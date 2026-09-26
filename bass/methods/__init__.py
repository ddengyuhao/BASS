from .bass_method import BASS
from .uniform_sampling import UniformSampling

METHOD_REGISTRY = {
    "BASS": BASS,
    "Uniform": UniformSampling,
}
