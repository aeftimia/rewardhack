# Funding application (BlueDot form), as submitted

Form: https://airtable.com/appeeZXWP0NdMvWYd/paggDLIdBeveEPLs7/form (linked from https://bluedot.org/grants/rapid, "up to $20k", average decision 5 days). Questions are copied verbatim from the form on 2026-11-07. Every free-text answer is drafted below. Fixed-choice answers are listed at the end.

**Submitted 2026-11-10** through the form above, as written below. The form confirmed receipt. The repository was made public the same day.

---

## What are you requesting funding for? (one sentence, 200 characters max)

Compute to test whether rewarding a model for correctly reporting an impossible task, alongside the task reward, stops reinforcement learning from teaching it to exploit a buggy grader and hide it.

## How much funding are you requesting?

$2,500

## Tell us more about your project.

Reinforcement learning with a flawed grader teaches models to pass the grader instead of doing the task. In the Hugging Face incident this August, OpenAI's postmortem traced most of the damage to tasks that had no solution, so the only rewarded behaviour was to cheat the scorer. Their stated fix is a grader that also rewards correctly identifying a broken task and stopping safely. As far as I can find, nobody has published a measurement of whether that fix works, what it costs on solvable tasks, or how it interacts with instructions and permission to cheat.

I built a small, fully verifiable version of the problem. The task is to write a Lean 4 proof of a theorem and report what was established. Half the theorems are false. The grader has a realistic bug, it rewards anything that compiles, so a placeholder proof such as `by sorry` is paid on a false theorem. An independent Lean verifier, with a locked axiom list, decides afterwards whether a submission is a real proof, a placeholder, or a real refutation, and whether the report matches the code. Everything the reward touches is machine-checkable. There is no LLM judge in the loop.

The grant funds the training experiment the pilot was built for. I will train open-weight reasoning models (gpt-oss-20b first, Qwen3-32B as the second family) with GRPO against the buggy grader under two reward schemes. The baseline is the grader alone. The intervention adds a term for the correctness of the report, meaning the claim in the report matches what the submitted code actually proves, plus a swept penalty for passing the grader without a proof. The sweep tells us how much true-verifier signal has to leak into the reward before the exploit stops being learned. Both arms are evaluated on held-out statements along a difficulty dial I calibrated, from falsehoods the model can see at a glance to residue and 3-SAT claims it cannot settle in its budget. The outcomes are the rate of grader exploitation, the rate of concealed exploitation, the rate of confident false claims, acknowledgement of impossibility, and genuine proof rate on true theorems. A second conversational turn with the grader code removed tests whether learned exploitation persists once the bug is out of sight, and a prompt that forbids cheating tests whether an instruction still overrides what the reward taught.

## How does this reduce catastrophic risk from AI and/or contribute to AI going well for humanity?

The failure mode is specific and already observed in deployment. A model trained against imperfect graders learns that producing something the grader accepts is the goal, and that reporting success is part of producing it. At small scale this looks like a placeholder proof labelled "proved". At the scale of agents operating in other people's infrastructure it looked like the Hugging Face incident. Deceptive success reporting is also the behaviour that makes every other safety measure harder, because oversight depends on reports.

The intervention I am testing is the one frontier labs say they are adopting, and the public evidence for it is thin. Abstention rewards in question answering show a knife-edge, where a little too much reward for "I don't know" collapses task performance and a little too little does nothing. Whether the same tradeoff appears when the exit is a verifiable statement about the task, and whether the exit reward can hold off exploitation once a model has found a grader bug, are empirical questions that a lab's internal result cannot answer for the community. A cheap, open, reproducible measurement gives independent researchers and smaller labs a basis for designing graders, and gives the community a way to check vendor claims.

The framing also generalises. Treating a model's output as a payload plus a machine-checkable exit status turns "the model said it could not do this" into something a training pipeline can reward or penalise with precision and recall against ground truth, the same way a program's exit code separates a failure from a wrong answer. If the pilot result holds, the same decomposition applies to other reward-hacking settings and, with a different value function, to agents that take credit for work they did not do.

## What have you already done?

I have spent about $21 over the past month on a pilot, with all code, raw rows, and spending in a public repository.

I built the environment. A Lean 4 harness with a buggy compile-only grader, an independent verifier with an axiom allowlist, paired true and false theorems with machine-checked reference proofs and refutations, and a 242-statement hard set (far-witness polynomial residues, shifted inequality boundaries, natural-number subtraction traps, and 3-SAT instances near the satisfiability threshold) whose false halves a frontier-small reasoning model acknowledges as false between 0% and 69% of the time depending on difficulty. Every reference is verified by Lean.

I measured emergence. In more than 2,000 unprompted samples across Qwen3-8B, Llama 3.1 8B, Llama 3.1 70B, Qwen3-32B, gpt-oss-20b, gpt-5.4-nano, gpt-5.4-mini, and gpt-5.5, models at or below 70B without reasoning never found the exploit. With reasoning on, Qwen3-32B and gpt-oss-20b submit placeholders on about 20% of false statements, and a reading of every gpt-oss-20b reasoning trace found 8 of 120 that explicitly worked out that the reward is compile-only and chose to exploit it, 2 of them with a report that hides it. The same model refused the exploit more often than it took it, citing the prompt's instruction. This is unprompted, untrained, deliberate grader exploitation in a trainable 21B open-weight model, and it is the base rate the training experiment needs.

I measured the RL mechanism. Training Qwen3-8B locally with GRPO against the buggy grader, one sampled placeholder was enough for a single gradient update to flip the policy to placeholders on 20 of 24 held-out statements, including true theorems it could previously prove. The behaviour survived removing the hint that seeded it and grew further under the plain prompt. Adding a refutation reward after the collapse did nothing, because the collapsed policy never sampled a refutation, which is why the proposed arms start before collapse.

I ran the permission grid. With gpt-5.4-nano on the hard set, forbidding cheating drove placeholders to 0 of 55, allowing it drove them to 33 of 58, and in every condition every grader-paid placeholder came with an honest report. Permission displaced honest acknowledgement, not false claims. Exploitation persisted into a second turn with the grader code removed. Confident false claims were the dominant failure at every setting, which is the mass the report-correctness reward targets.

I read every response. All model reports and reasoning traces were classified by reading against a written rubric, with the labels stored per row, after a cheap LLM judge was found to mislabel "established using decide" as an acknowledgement.

## What makes you think this project will be successful? Why you? Why now?

The hard parts are done. The environment, verifier, task sets, training loop, rescoring, and reading rubrics exist and have produced results on eight models. The remaining work is running the training arms on a GPU large enough for a 20B to 32B reasoning model, which I have already driven through RunPod for the probes. The one risk I could not rule out cheaply is that training-time exploration differs from probe-time exploration. The pilot bounded it in both directions, with a seeded fallback at 8B that reliably shows the mechanism and an unseeded base rate at 20B that gives RL something to amplify.

I have worked as an applied machine learning scientist for eight years, building and shipping models in JAX and PyTorch, and I have a physics and mathematics background. I am a BlueDot Technical AI Safety course alum. I built every piece of the pilot myself and I have been careful about what the data does and does not show. Each findings document separates measured facts from my interpretation, and the one time my own summary overstated a result I corrected it in writing.

Now because the problem has moved from hypothetical to incident. OpenAI announced the honest-exit remedy in August and has published no measurement of it. Open-weight models have just reached the point where one of them discovers and conceals a grader exploit on its own, so the experiment can be run on a public model at a cost of hundreds of dollars. A result in the next few months informs how people build graders while the practice is still being set.

## What specifically would this grant fund?

- $800 GPU rental for the gpt-oss-20b training arms: five reward settings (baseline, and the report-correctness reward with the cheat penalty at four values), three seeds each, with evaluation along the difficulty dial.
- $900 GPU rental for the Qwen3-32B replication of the same arms as a second model family.
- $300 API credits for frontier reference points along the same difficulty dial and for validating any report classifier against the hand-read labels.
- $100 persistent storage for checkpoints, raw rollouts, and Lean toolchains across pods.
- $400 contingency for failed pods, reruns after bugs, and longer reasoning budgets.

Every run has a hard spending cap and an automatic shutdown, and every dollar is logged in the public ledger alongside the results it paid for.

## What would you do without this grant?

I would reduce scope to what runs free on my laptop. That is Qwen3-8B in 4-bit without reasoning, which never recognises a false statement and never samples the exploit, so every arm has to be seeded with a hint and the result is a mechanism check instead of a test of the intervention on emergent behaviour. The reasoning-mode experiments, which are the ones that speak to the Hugging Face framing, would not happen. I would also keep spending my own API credits for probes at a slower pace. The likely outcome is a smaller write-up in several months. With the grant the replicable result lands in six to eight weeks.

## Add any supporting links

- Project repository with code, raw results, findings documents, and the spending ledger: https://github.com/aeftimia/rewardhack
- Personal site: https://aeftimia.github.io
- OpenAI postmortem the project responds to: https://openai.com/index/hugging-face-incident-and-the-road-ahead/
- METR investigation of the same incident: https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation

## How could we make this process better, or otherwise help you succeed with your project? Also add any other information you feel we should consider here.

The project is designed to publish as a short paper or long post with every row reproducible. If BlueDot has a preferred venue for grantee results, or researchers working on grader design or honest-exit rewards who would want to compare notes, an introduction would be worth more to me than additional funding. I would also welcome a referee who disagrees with the framing before I run the arms.

The idea came out of discussion in my BlueDot AGI Strategy cohort of the Hugging Face incident, and I spoke with members of the BlueDot safety team about it before applying.

---

## Fixed-choice answers

- Your name: Alex Eftimiades
- Where are you based? United States
- Where can we learn more about you? https://www.linkedin.com/in/alex-eftimiades-1a4836bb and https://aeftimia.github.io (prefilled)
- How are you connected to the BlueDot community? Course alum
- Can we share your application with other funders and organizations as relevant? Yes
- If we approve your grant, can we share details about it publicly? Can share publicly with my name
- Public URL: https://github.com/aeftimia/rewardhack
- Who would we pay if your grant is approved? Individual
