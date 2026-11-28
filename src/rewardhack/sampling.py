"""The one sampling setting every generation path uses: training rollouts, evaluations and benchmarks.

Plain temperature 1 over the whole vocabulary (owner, 26 Nov, after the review of notes section 72): the usual setting
for RL on language models and gpt-oss's own recommendation. Truncation (the earlier top_p 0.95, top_k 20) removes the
rare tokens where an exploit is first sampled, and needs vLLM's per-token sampling mask for a consistent importance
ratio. In every library used here top_p 1.0 and top_k 0 both mean "no truncation".
"""
DEFAULT_SEED = 42
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
SAMPLING_KWARGS = dict(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
