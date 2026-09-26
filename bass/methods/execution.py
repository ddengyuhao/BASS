from collections import defaultdict

from ..utils import extract_answer_from_text, format_options


GROUNDING_PROMPT = (
    "The images are representative frames of one event ({start:.1f}s - {end:.1f}s) "
    "from a long video.\n\n"
    "Question: {question}\n"
    "Options:\n{options}\n\n"
    "Write an evidence record for this event. List only entities, actions, and "
    "attributes that are clearly visible in the frames and that may help answer the "
    "question or link this event to other events in the video (e.g., distinctive "
    "people, objects, clothing, text, or locations). Omit details unrelated to the "
    "question and anything that cannot be verified from the frames. "
    "Use short bullet points and do not answer the question."
)

VERIFICATION_PROMPT = (
    "Two events from the same video are described below. Event 1 occurs before Event 2.\n\n"
    "Question: {question}\n\n"
    "Event 1 ({t1}):\n{record1}\n\n"
    "Event 2 ({t2}):\n{record2}\n\n"
    "Do these two records support a consistent relation between the events that is "
    "relevant to the question (for example, the same person, object, or place appears "
    "in both, or Event 1 leads to Event 2)? Reply with 'Yes' or 'No' on the first line. "
    "If 'Yes', describe the relation in one sentence on the second line."
)

SYNTHESIS_PROMPT = (
    "The following evidence was extracted from a long video. Events are listed in "
    "temporal order, and the relation between consecutive events is given in brackets.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Options:\n{options}\n\n"
    "Answer the question based only on the evidence above. "
    "End your response with 'The answer is X.'"
)


COT_PROMPT = (
    "The images are frames from selected events of a long video, in temporal order:\n"
    "{timeline}\n\n"
    "Question: {question}\n"
    "Options:\n{options}\n\n"
    "Let's think step by step. End your response with 'The answer is X.'"
)


class GraphGuidedExecutor:
    """
    Graph-guided execution over the selected event set S (Sec. 4.2):
      Phase 1  event grounding     -> evidence record d_v per event
      Phase 2  relation validation -> keep supported semantic edges of G[S]
      Phase 3  result synthesis    -> weighted vote over directed reasoning paths (Eq. 10)
    """

    def __init__(self, model, max_new_tokens=512, max_paths=8, max_path_enum=1000):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.max_paths = max_paths
        self.max_path_enum = max_path_enum

    def run(self, question, options, selected, events, event_frames, semantic_edges):
        """
        Args:
            selected: selected event ids S.
            events: list of (start_sec, end_sec) for all events.
            event_frames: dict event id -> list of PIL images (representative frames).
            semantic_edges: dict (i, j) -> weight of the full event graph.
        Returns:
            answer letter (or None) and an execution trace.
        """
        selected = sorted(selected)
        options_str = format_options(options)
        num_options = len(options) if isinstance(options, (list, tuple)) else 4

        # Phase 1: event grounding (one record per event, reused by all edge checks)
        records = {}
        for v in selected:
            start, end = events[v]
            prompt = GROUNDING_PROMPT.format(
                start=start, end=end, question=question, options=options_str)
            records[v] = self.model.generate(
                event_frames[v], prompt, max_new_tokens=self.max_new_tokens).strip()

        # Induced subgraph G[S]
        in_s = set(selected)
        temporal = [(u, u + 1) for u in selected if u + 1 in in_s]
        candidates = sorted((i, j) for (i, j) in semantic_edges if i in in_s and j in in_s)

        # Phase 2: relation validation of semantic edges
        relations = {}
        for (u, v) in candidates:
            prompt = VERIFICATION_PROMPT.format(
                question=question,
                t1=self._time(events[u]), record1=records[u],
                t2=self._time(events[v]), record2=records[v])
            reply = self.model.generate_text(prompt, max_new_tokens=self.max_new_tokens)
            keep, statement = self._parse_verification(reply)
            if keep:
                relations[(u, v)] = statement

        edges = {e: 1.0 for e in temporal}
        edges.update({e: semantic_edges[e] for e in relations})

        # Phase 3: result synthesis over directed reasoning paths
        paths = self._enumerate_paths(selected, edges)
        if not paths:
            # No retained edge: use all records in temporal order as one context
            paths = [(tuple(selected), 1.0)]

        votes = defaultdict(float)
        path_trace = []
        for path, score in paths:
            context = self._path_context(path, events, records, relations)
            prompt = SYNTHESIS_PROMPT.format(
                context=context, question=question, options=options_str)
            reply = self.model.generate_text(prompt, max_new_tokens=self.max_new_tokens)
            ans = extract_answer_from_text(reply, num_options=num_options)
            if ans is not None:
                votes[ans] += score
            path_trace.append({"path": list(path), "score": score, "answer": ans})

        answer = max(votes, key=votes.get) if votes else None
        trace = {
            "selected": selected,
            "records": {str(k): v for k, v in records.items()},
            "verified_edges": [[u, v, s] for (u, v), s in relations.items()],
            "rejected_edges": [[u, v] for (u, v) in candidates if (u, v) not in relations],
            "paths": path_trace,
            "votes": dict(votes),
        }
        return answer, trace

    def run_cot(self, question, options, selected, events, event_frames):
        """
        Standard Chain-of-Thought baseline (Sec. 5.3, Fig. 6): the frames of the same
        selected events are given to the LMM in a single call, without grounding,
        relation validation, or path aggregation.
        """
        selected = sorted(selected)
        num_options = len(options) if isinstance(options, (list, tuple)) else 4
        frames, timeline = [], []
        for k, v in enumerate(selected):
            frames.extend(event_frames[v])
            timeline.append(f"- Event {k + 1}: {self._time(events[v])}, "
                            f"{len(event_frames[v])} frame(s)")
        prompt = COT_PROMPT.format(
            timeline="\n".join(timeline), question=question, options=format_options(options))
        reply = self.model.generate(frames, prompt, max_new_tokens=self.max_new_tokens)
        answer = extract_answer_from_text(reply, num_options=num_options)
        return answer, {"selected": selected, "response": reply}

    def _enumerate_paths(self, nodes, edges):
        """
        Maximal directed paths (more than one event) over the retained edges.
        All edges point forward in time, so the graph is a DAG.
        Returns the top `max_paths` paths by average edge weight.
        """
        succ = defaultdict(list)
        has_in = set()
        for (u, v) in edges:
            succ[u].append(v)
            has_in.add(v)
        for u in succ:
            succ[u].sort()

        paths = []

        def dfs(path):
            if len(paths) >= self.max_path_enum:
                return
            nxt = succ.get(path[-1], [])
            if not nxt:
                if len(path) > 1:
                    w = [edges[(path[k], path[k + 1])] for k in range(len(path) - 1)]
                    paths.append((tuple(path), sum(w) / len(w)))
                return
            for v in nxt:
                dfs(path + [v])

        for s in nodes:
            if s in succ and s not in has_in:
                dfs([s])

        paths.sort(key=lambda p: (-p[1], p[0]))
        return paths[:self.max_paths]

    def _path_context(self, path, events, records, relations):
        lines = []
        for k, v in enumerate(path):
            lines.append(f"Event {k + 1} ({self._time(events[v])}):\n{records[v]}")
            if k + 1 < len(path):
                nxt = path[k + 1]
                if (v, nxt) in relations:
                    lines.append(f"[Relation: {relations[(v, nxt)]}]")
                else:
                    lines.append("[Next in time]")
        return "\n".join(lines)

    @staticmethod
    def _parse_verification(reply):
        lines = [l.strip() for l in (reply or "").strip().splitlines() if l.strip()]
        if not lines or not lines[0].lower().startswith("yes"):
            return False, ""
        statement = lines[1] if len(lines) > 1 else lines[0][3:].strip(" .,:-")
        return True, statement

    @staticmethod
    def _time(event):
        return f"{event[0]:.1f}s - {event[1]:.1f}s"
