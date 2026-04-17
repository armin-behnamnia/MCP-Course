# **Gradient Regularization Prevents Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards** 

**Johannes Ackermann**[1 2] **Michael Noukhovitch**[3 4] **Takashi Ishida**[2 1] **Masashi Sugiyama**[2 1] 

## **Abstract** 

Reinforcement Learning from Human Feedback (RLHF) or Verifiable Rewards (RLVR) are two key steps in the post-training of modern Language Models (LMs). A common problem is reward hacking, where the policy may exploit inaccuracies of the reward and learn an unintended behavior. Most previous works address this by limiting the policy update with a Kullback-Leibler (KL) penalty towards a reference model. We propose a different framing: Train the LM in a way that biases policy updates towards regions in which the reward is more accurate. First, we derive a theoretical connection between the accuracy of a reward model and the flatness of an optimum at convergence. Gradient regularization (GR) can then be used to bias training to flatter regions and thereby maintain reward model accuracy. We confirm these results by showing that the gradient norm and reward accuracy are empirically correlated in RLHF. We then show that Reference Resets of the KL penalty implicitly use GR to find flatter regions with higher reward accuracy. We further improve on this by proposing to use explicit GR with an efficient finite-difference estimate. Empirically, GR performs better than a KL penalty across a diverse set of RL experiments with LMs. GR achieves a higher GPT-judged win-rate in RLHF, avoids overly focusing on the format in rule-based math rewards, and prevents hacking the judge in LLM-as-a-Judge math tasks. 

## **1. Introduction** 

Reinforcement Learning (RL) has become a key part of the post-training of language models (LMs) (Stiennon et al., 2020; Shao et al., 2024). In the case of RL from Human Feedback (RLHF) (Christiano et al., 2017), we use RL to 

> 1The University of Tokyo 2RIKEN AIP 3Mila 4Université de Montréal. Correspondence to: Johannes Ackermann <ackermann@ms.k.u-tokyo.ac.jp>. 

align the behavior of LMs with human preferences, which we cannot easily represent with a rule-based reward. In the case of RL from Verifiable Feedback (RLVR) (Havrilla et al., 2024; Lambert et al., 2024), RL is used to improve the performance on tasks with verifiable rewards, such as mathematical reasoning or agentic tasks. In RLHF, we use pairwise comparison data to train a reward model (RM), which then provides the reward estimates for policy updates. In RLVR, we use a verifier such as a rule-based reward or another Large Language Model (LLM) to check if the model output matches the expected answer. In both cases there is a desired behavior, corresponding to a desired true reward, which we try to approximate with the trained RM, rulebased reward, or LLM-as-a-judge reward. We collectively refer to them as proxy rewards (PRs). A key challenge of RL post-training is: How can we ensure that when updating our policy with the PR, we actually improve the true reward, i.e., how can we ensure that our PR stays accurate as the policy changes throughout training? One solution is to iteratively update the PR during training with new data from the current policy (Christiano et al., 2017). As this can be costly, another option is to use a Kullback-Leibler (KL) penalty to ensure the policy stays close to the initial model (Stiennon et al., 2020). In practice, the KL penalty slows down training and may not even improve performance (Gao et al., 2023), leading to recent papers abandoning it for tasks with rule-based rewards (GLM-4.5 Team et al., 2025; Olmo Team et al., 2025). However, this risks reward hacking with reward models and LLM-as-a-judge. 

We thus aim to modify the policy update such that the policy not only maximizes the PR, but also maximizes the PR accuracy, without constraining it to stay close to the initial policy. We argue that reward hacking often corresponds to sharp optima of the PR and propose gradient regularization (GR) as a solution. We illustrate an overview in Figure 1. 

In Section 3, we formalize this notion and show a theoretical connection between the flatness of an optimum and the PR accuracy at this optimum, as measured by the Bradley-Terry (BT; Bradley & Terry, 1952) loss. In Section 4, we show that this theoretical connection can be utilized to improve RLHF in practice. We leverage a recent method, Reference Resets (Liu et al., 2025a), and demonstrate that it implicitly 

_Preprint. February 23, 2026._ 

1 

**Gradient Regularization Prevents Reward Hacking** 

**==> picture [481 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
The answer is}}}))})))}3} The answer is 4. 0 . 9 0 . 4 No GRGR<br>1.0<br>0 . 8 0 . 2<br>The answer is 3.<br>0 . 7 0<br>0 500 1000 0 500 1000 0 500 1000<br>Gradient Step Gradient Step Gradient Step<br>Reward Proxy Reward True Reward Gradient Norm<br>**----- End of picture text -----**<br>


_Figure 1._ We argue that reward hacking often corresponds to exploiting sharp maxima in action space, as illustrated by the conceptual figure (left). For example, an LLM judge may be confused and assign a high reward to a wrong answer with specific formatting. In the LLM-as-a-Judge training run shown on the right, the increase in gradient norm coincides with reward hacking, resulting in true reward collapsing. By using gradient norm regularization, we can prevent this issue and obtain a better model, as seen by the improved true reward. The examples show Qwen2.5-0.5B models trained on GSM8K with a Qwen2.5-1.5B-Instruct judge with access to the true answer. 

regularizes the gradient, providing a novel interpretation of this method. We empirically show this enables better RLHF performance on the TL;DR task. In Section 5, we propose a novel application of explicit gradient regularization to RL post-training. We demonstrate that it outperforms baselines in RLHF tasks with reward models as well as rule-based and LLM-as-a-judge RLVR math tasks. By using GR, we are able to train proficient LLM-as-a-Judge and RLHF models _completely replacing the standard KL penalty_ . 

## **2. Background** 

2025b), which removes normalization terms from GRPO. As we do not do multiple updates per batch, Dr.GRPO simplifies to using REINFORCE with the baseline being the sample average over multiple actions drawn for the same state, i.e, _b_ ( _s_ ) = _N_ 1 � _Ni_ =1 _[R]_[�][(] _[s, a][i]_[)][,][with] _[a][i][∼] πϕ_ ( _a|s_ ). GRPO was originally presented with a KL penalty _D_ KL( _πϕ_ ; _πϕ_ 1), as commonly used in RLHF (Stiennon et al., 2020). This KL-penalty, weighted by a hyper-parameter _β_ , is intended to keep the model close to the initial policy _πϕ_ 1 , preventing reward hacking (Stiennon et al., 2020), but is sometimes omitted in more recent works (GLM-4.5 Team et al., 2025; Olmo Team et al., 2025). 

## **2.1. Reinforcement Learning** 

As is common in the RL post-training literature (Shao et al., 2024; Ahmadian et al., 2024), we consider an episode consisting of a single state _s_ , representing the prompt, and a single action _a_ , representing the reply. We denote the state set by _S_ and the action set by _A_ . As policy _πϕ_ with parameters _ϕ_ , we consider an autoregressive LM with conditional probability _πϕ_ ( _a|s_ ) =[�] _t[π][ϕ]_[(][a] _[t][|][s,]_[ a] _[<t]_[)][, where][ a] _[t]_ is the _t_ -th response token, a _<t_ refers to the response tokens before a _t_ and _a_ refers to the entire response. We assume that there is a true reward function _R[∗]_ : _S × A →_ R and our goal is to maximize the return of the policy _πϕ_ on this reward _J[∗]_ ( _ϕ_ ) = E _s∼P_ ( _s_ ) _,a∼πϕ_ [ _R[∗]_ ( _s, a_ )]. However, we do not have access to _R[∗]_ . Instead, we use a proxy reward (PR) _R_[�] , and train the policy to maximize its return _J_ ( _ϕ, θ_ ) = E _s∼P_ ( _s_ ) _,a∼πϕ_ [ _R_[�] _θ_ ( _s, a_ )]. To optimize the return we use REINFORCE policy gradient updates (Williams, 1992) 

**==> picture [227 x 34] intentionally omitted <==**

where _b_ ( _s_ ) is a baseline. In this work, we use a variant of Group Relative Policy Optimization (GRPO) (Shao et al., 2024) called GRPO Done Right (Dr.GRPO) (Liu et al., 

## **2.2. Proxy Rewards** 

We consider three types of PRs: Trained reward models _R[θ]_ , rule based rewards _R_[R] , and LLM-as-a-Judge rewards _R_[LM] . 

**Reward Models** (RMs) are commonly used in RLHF (Christiano et al., 2017; Stiennon et al., 2020) to represent complex preferences which are hard to turn into rule-based rewards. They generally assume the Bradley-Terry (BT) model of preference (Bradley & Terry, 1952) where the probability _P_ of preferring one option _a_ 1 over another option _a_ 0 is the logistic function _σ_ ( _x_ ) = 1 _/_ (1 + _e[−][x]_ ) of the difference of the true reward for each action, i.e., 

**==> picture [212 x 11] intentionally omitted <==**

Pairs of responses ( _a_ 0 _, a_ 1) for each prompt _s_ are collected from an initial model _πϕ_ 1. Human annotators then choose their preferred (winning) reply _a_ w and not preferred (losing) reply _a_ l, creating a dataset _D_ RM = _{_ ( _s[j] , a[j]_ w _[, a][j]_ l[)] _[}][N] j_ =1[RM][.][We] can then use the BT assumption to train an RM _Rθ_ : _S × A →_ R with parameters _θ ∈_ Θ by minimizing the crossentropy loss based on the BT model: 

**==> picture [224 x 29] intentionally omitted <==**

2 

**Gradient Regularization Prevents Reward Hacking** 

where _Pπϕ_ = _P_ ( _s_ ) _πϕ_ ( _a_ 0 _|s_ ) _πϕ_ ( _a_ 1 _|s_ ) _P_ ( _a_ 1 _> a_ 0 _|s_ ) is the probability of the ( _s, a_ w _, a_ l) triplet under the policy _πϕ_ . To train the RM we minimize _L_ BT( _θ, ϕ_[1] ) and the expectation is replaced by the sample average from _D_ RM. 

**Rule-Based Rewards** (Havrilla et al., 2024) are deterministic checks whether a final answer matches the ground truth. To encourage reasoning, and to allow a discrimination of a final answer against occurrences of the answer during reasoning, rule-based rewards typically require the answer to follow a specific format, for example using the LaTeX tag \boxed{} (Hendrycks et al., 2021) or the HTML tags (DeepSeek-AI, 2025) <think>...</think><answer>...</answer>. Thus, oftentimes one reward term _R_[F] checks whether the format matches and another reward term _R_[C] checks for correctness, to give the rule-based reward _R_[R] = _R_[C] + _R_[F] . 

**LLM-as-a-Judge** (Zheng et al., 2023) prompts an LLM and uses its textual output to check whether an answer is correct. Frequently designing a rule-based reward can be challenging due to many possible accurate solutions, e.g., the correctness of a proof or many equivalent ways to write a math answer. Instead, prompting an LLM-as-a-Judge with a description of scoring criteria, the question, and the correct answer (if available) can allow us to capture solutions more robustly. LLM-as-a-Judge can also allow for more complicated rewards with a combination of objective (e.g., correct reply) and subjective (e.g., clear reasoning) criteria. 

## **2.3. Gradient Regularization (GR)** 

Flat minima of a loss function _L_ ( _ϕ_ ) are connected to better generalization in supervised learning (Hochreiter & Schmidhuber, 1997; Foret et al., 2021), i.e., a smaller difference between the population/test loss _L_ ( _ϕ_ ) = E( _x,y_ ) _∼P_ ( _x,y_ )[ _ℓ_ ( _fϕ_ ( _x_ ) _, y_ )] and its empirical approximation on a finite training dataset _D_ , consisting of i.i.d. input-label pairs ( _x, y_ ) _∼ P_ ( _x, y_ ), for a loss function _ℓ_ . A way to obtain flat minima is by regularizing the gradient norm of the objective _L_ , i.e., the squared Euclidean norm _∥∇ϕL_ ( _ϕ_ ) _∥_[2] (Zhao et al., 2022). Adding this term to our loss function, we need to calculate its gradient, which can be approximated with a parameter perturbation (Karakida et al., 2023) 

**==> picture [226 x 31] intentionally omitted <==**

The model parameters are then updated as 

**==> picture [205 x 19] intentionally omitted <==**

where _η ∈_ R[+] is the learning rate and _γ ∈_ R is a hyperparameter controlling the strength of the GR and _ε_ controls the strength of the parameter perturbation. 

## **3. Accurate Proxy Rewards via Gradient Regularization** 

## **3.1. Problem Formulation and Overview** 

As mentioned above, the goal of RL is to learn a policy _πϕ_ that maximizes the expected true reward _R[∗]_ . As we do not have access to _R[∗]_ we instead have to use the PR _R_[�] to update our policy. As the PR is generally an approximation of the true reward _R[∗]_ , it may be prone to reward hacking and we need to ensure that the PR stays accurate during training. For PRs that have been trained or designed based on samples from an initial policy _π_[1] , such as RMs in RLHF, the most common solution is to use a KL penalty (Stiennon et al., 2020; Shao et al., 2024) ensuring that the policy stays close to _π_[1] and thus the PR stays accurate. However, the KL penalty also limits how much the policy can learn. Instead of changing the PR _R_[�] or constraining _πϕ_ , we aim to update the policy _πϕ_ such that it obtains a high reward provided by the PR _R_[�] , while also biasing the policy update towards regions in which the PR _R_[�] is accurate. Our goal is thus 

**==> picture [192 x 19] intentionally omitted <==**

where _γ >_ 0 is a hyper-parameter and we use the BT loss _L_ BT( _θ, ϕ_ ) of a PRs parameterized with _θ_ on actions sampled from _πϕ_ . However, we cannot directly evaluate the BT loss _L_ BT( _θ, ϕ_ ), as it requires pairwise comparisons for actions drawn from _πϕ_ . As we will show below, under some assumptions, overly sharp optima of the objective _J_ ( _ϕ, θ_ ) correspond to overly sharp maxima of the PR _R_[�] , which imply an excess BT loss _L_ BT of the PR. Regularizing the policy gradient norm during training biases the optimization towards flat optima (Zhao et al., 2022), avoiding this problem. Instead of _L_ BT( _θ, ϕ_ ), we thus use the policy gradient norm _∥∇ϕJ_ ( _ϕ, θ_ ) _∥_[2] , yielding the practically optimizable 

**==> picture [202 x 19] intentionally omitted <==**

## **3.2. Gradient Regularization Improves Proxy Reward Accuracy** 

Our argument consists of three steps: GR biases optimization towards flat maxima. Flat maxima imply pairwise robust policies. Pairwise robust policies correspond to accurate PRs, under the assumption of a flat reward. We provide an illustration in Figure 2. We note that our argument in this section focuses on continuous action spaces, while LMs use discrete action spaces. We discuss this discrepancy at the end of this section and provide experimental evidence in the LM setting in Section 4 and Section 5. 

**GR Favors Flat Maxima in Parameter Space** It is well known that GR favors flat minima in the parameter space 

3 

**Gradient Regularization Prevents Reward Hacking** 

**==> picture [427 x 78] intentionally omitted <==**

**----- Start of picture text -----**<br>
J ( ϕ, θ ) R �( s, a ) D<br>sharp flat flat L - D  region<br>L too sharp<br>πϕ sharp πϕ flat<br>ϕ a<br>**----- End of picture text -----**<br>


_Figure 2._ Conceptual illustration of our theoretical argument: (left) Regularizing the gradient norm biases optimization toward flat basins in parameter space, and (right) under action-smoothness, a flat maximum makes _δ_ -close pairs unlikely to have a reward gap larger than _K_ , i.e. decreases the probability of overly sharp action pairs _a_ 1 _, a_ 2 : _∥a_ 1 _− a_ 2 _∥≤ δ, |R_[�] ( _s, a_ 1) _− R_[�] ( _s, a_ 2) _| > K_ . Under the assumption of a Lipschitz-continuous true reward _R[∗]_ , each such pair implies an incorrect proxy reward _R_[�] . 

(Barrett & Dherin, 2021; Karakida et al., 2023). One way to see this is the argument of Zhao et al. (2022), connecting the gradient norm to Lipschitz continuity, which we reproduce for completeness in Appendix B.3. It is also well known in the supervised learning literature (Hochreiter & Schmidhuber, 1997) that flat minima lead to better generalization, i.e., a smaller difference between the population loss _L_ ( _θ_ ) and the loss on the training set _L_[�] ( _θ_ ). Equivalently, in RL, we would directly expect GR to ensure� a similar PR score obtained on the training prompts _J_ ( _ϕ, θ_ ) = E _s∼D_ tr _,a∼πϕ_ [ _R_ � _θ_ ( _s, a_ )] and on the test prompts _J_ ( _ϕ, θ_ ) = E _s∼D_ te _,a∼πϕ_ [ _R_[�] _θ_ ( _s, a_ )]. Instead of considering generalization, we consider the PR accuracy, or more precisely the BT loss _L_ BT, necessitating the next two steps. 

**Flat Maxima in Parameter Space imply Robust Maxima in Action Space** We need to connect the flatness of an optimum in parameter space with its flatness in action space, often referred to as robustness. Lee & Yoon (2025) previously investigated this connection under the assumption of a ball with constant reward, which we use as starting point. While they assume a constant return on a ball _B_ ( _ϕ[∗] , E_ ) := _{ϵ_ : _∥ϵ∥≤E}_ around the optimum _ϕ[∗]_ , we allow the reward to decrease by at most _L_[ˆ] : 

**Definition 3.1** ( _E − L_[�] flat reward maximum) **.** For a reward function _R_ ( _s, a_ ) and policy _πϕ_ ( _a|s_ ), parameterized by _ϕ_ , a maximum _ϕ[∗]_ is _E − L_[�] -flat if the following holds: 

**==> picture [227 x 34] intentionally omitted <==**

We also define the concept of a ( _δ, K, ρ_ )-pairwise robust policy, which measures the probability of action pairs ( _a_ 1 _, a_ 2) sampled from the policy violating _K/δ_ -Lipschitz-continuity, i.e. _|R_[�] ( _s, a_ 1) _− R_[�] ( _s, a_ 2) _| > K, ∥a_ 1 _− a_ 2 _∥ < δ_ : 

**Definition 3.2** (( _δ, K, ρ_ )-pairwise robust policy) **.** For a policy _πϕ_ ( _a|s_ ) and a reward _R_ ( _s, a_ ), define the sharpness set _SK,δ_ ( _R_ ) := _{_ ( _s, a_ 1 _, a_ 2) : _∥a_ 1 _− a_ 2 _∥≤ δ, |R_ ( _s, a_ 1) _− R_ ( _s, a_ 2) _| > K}_ , i.e. the set of _δ_ -close action pairs for 

which the reward changes by more than _K_ . A policy _πϕ_ is ( _δ, K, ρ_ )-pairwise robust for a reward _R_ if 

**==> picture [163 x 11] intentionally omitted <==**

with _P_ ( _SK,δ_ ( _R_ ) _|πϕ_ ) being the probability under _s ∼ P_ ( _s_ ) i _._ i _._ d _._ and ( _a_ 1 _, a_ 2) _∼ πϕ_ ( _a|s_ ). 

We obtain the following proposition, linking flatness in parameter space to ( _δ, K, ρ_ ) in action space, under the assumption of a _β_ -smooth PR: 

**Proposition 3.3** ( _E − L_[�] flat reward implies ( _δ, K, ρ_ ) robust policy) **.** _Assume a Gaussian policy with fixed covariance_ Σ _, where we denote the policy noise Z ∼N_ (0 _,_ Σ) _, and that R_[�] ( _s, a_ ) _is β-smooth in a. Further,_ J( _ϕ[∗]_ ) := _∇ϕ µϕ_ ( _s_ ) �� _ϕ_ = _ϕ[∗][is][the][Jacobian][matrix][of][the][mean][ac-] tion µϕ_ ( _s_ ) _. If ϕ[∗] is an E − L_[�] _flat reward maximum, then the PR action gradient ∥∇aR_[�] ( _s, µϕ_ ( _s_ )) _∥ is bounded by_ 

**==> picture [129 x 25] intentionally omitted <==**

_with radius D[∗] ≤∥_ J( _ϕ[∗]_ ) _∥E_ + _O_ ( _E_[2] ) _. For a given δ >_ 0 _and K >_ 0 _, with K/δ > G, we then know that no pairwise K violations can occur within the radius r_ := _β_[1] � _δ[−][G]_ � _, and thus the policy ϕ[∗] is_ ( _δ, K, ρ_ ) _-robust with_ 

**==> picture [171 x 14] intentionally omitted <==**

The proposition follows first linking flatness in parameter space to flatness in action space (Lee & Yoon, 2025). Then, under a _β_ -smooth PR, by a gradient bound we can ensure the absence of non-Lipschitz action pairs within a radius _r_ around the mean. Thus a violation requires at least one of the actions to fall outside this region Pr( _∥Z∥ > r_ ) and the final results follows from a union bound. A full derivation is shown in Appendix B. Intuitively, for a given maximum, as the sensitivity to disturbances _L_[ˆ] increases, the probability of overly sharp actions _P_ ( _SK,δ|πϕ_ ) increases as well. As the radius of the robustness _E_ increases, _P_ ( _SK,δ|πϕ_ ) decreases 

4 

**Gradient Regularization Prevents Reward Hacking** 

or stays constant, as we can freely pick a smaller _E[′] < E_ . Thus, flatter, wider minima decrease the risk of sharp action pairs. Next we will show that a larger _P_ ( _SK,δ|πϕ_ ) incurs a larger excess BT loss _L_ BT. 

**Non-Robust Policies Imply Inaccurate Proxy Reward** To connect ( _δ, K, ρ_ )-robustness and the BT loss, we make the assumption of an _L_ -Lipschitz true reward, _|R[∗]_ ( _s, a_ 1) _− R[∗]_ ( _s, a_ 2) _| ≤ L∥a_ 1 _− a_ 2 _∥_ . We then obtain 

**Proposition 3.4.** _For prompts s ∼ P_ ( _s_ ) _, pairs of actions_ ( _R_ � _a, policy_ 1 _, a_ 2) _, L π-Lipschitz true reward functionϕ, and K > Lδ, the excess BT loss can be lower R[∗] , proxy reward bounded as_ 

**==> picture [234 x 24] intentionally omitted <==**

The proof is shown in Appendix B.2. A non-robust policy in action space at least incurs an excess BT loss proportionate to the probability of overly sharp pairs _P_ ( _SK,δ_ ) and the magnitude of the sharpness ( _σ_ ( _K_ ) _− σ_ ( _Lδ_ ))[2] . By changing the policy _ϕ_ to decrease the ratio of violating pairs _P_ ( _SK,δ_ ) or the magnitude of the violations _K_ , we obtain a policy that induces a smaller excess BT loss. As shown above, we can bias the policy updates towards such policies with GR. 

**Limitations** While GR itself can be applied to LMs regardless of whether the action space _A_ is discrete or continuous, our theoretical argument assumes a Gaussian policy and requires a distance function, which is difficult to define for LMs. As we assume the true reward to be Lipschitz under this distance, a distance under a representation that captures semantic closeness _ϕ_ ( _a_ ) : _A →_ R _[d]_ , such as the hidden space of the LM we are training, would be an appealing option. For example in the illustrative Figure Figure 1 (left), the distance in action corresponds to semantic similarity, not formatting. Further, we only address excess BT loss incurred by overly sharp maxima, i.e. overly sharp PR maxima. We do not show whether or not GR prevents convergence to flat but incorrect regions of the PR. 

To empirically validate our theory, we next show that implicit GR, via Reference Resets, can improve RLHF. In Section 5 we leverage explicit GR for further improvements. 

## **4. Reference Reset as Gradient Regularization** 

Instead of performing explicit GR, we can rely on the implicit GR inherent to stochastic gradient descent (Barrett & Dherin, 2021). We propose to leverage Reference Resets (Liu et al., 2025a) where the KL penalty is changed by iteratively resetting its reference to the current policy _πϕ_ during training i.e., the update is penalized with _D_ KL( _πϕ_ ; _πϕ′_ ), where every _R_ steps we set _ϕ[′] ← ϕ_ . Since implicit GR usually occurs during the later stages of training, we find 

Reference Resets with a sufficient number of gradient steps per iteration to be an effective way of obtaining flat maxima. While Liu et al. (2025a) reset the policy when the reward has stagnated, we instead choose to perform resets every _R_ gradient steps similar to prior work using resets in RL (Nikishin et al., 2022; Noukhovitch et al., 2023). We find that it is important to train beyond reward stagnation, as the gradient norm decreases significantly only after stagnation. 

**Setup** We experiment on the well-known TL;DR summarization task (Stiennon et al., 2020) of Reddit posts with human summaries. Following (Gao et al., 2023; Tang et al., 2024) we make this a controlled synthetic setup, where the preference and evaluation data is relabeled by a “gold” reward model, Skywork-RewardLlama-3.1-8B-v0.2 (Liu et al., 2024). Compared to noisy human preferences, this setup gives us an oracle that enables consistent evaluations. We run experiments with models from both Pythia (Biderman et al., 2023) and Qwen 2.5 (Qwen et al., 2025) families for different scales. See full details in Appendix C.2. 

**Gradient Norm Tracks Sharpness and RM Accuracy** To test our theory of GR, we plot the gradient norm against three important empirical values: 1) The training reward from the RM. 2) The sharpness of the current policy parameters _ϕ_ , which we predict is tied to reward-hacking. The sharpness is estimated by sampling 32 perturbations _{ϵi}_[32] _i_ =1 and evaluating _S_ ( _ϕ, θ_ ) = max _i J_ ( _ϕ, θ_ ) _− J_ ( _ϕ_ + _ϵi, θ_ ), for each checkpoint. And 3) the BT loss _L_ BT( _θ, ϕ_ ), which represents how accurate our reward model is for our current policy. We sample completions from our model, label with the gold RM, and get the BT loss under our training RM. In this way, we can empirically check whether we are training in a regime where our PR is accurate. We train a Pythia 1B model with GRPO+Reference Resets and show results in Figure 3 with dashed vertical lines representing resets. In each iteration the gradient norm initially spikes and the reward increases quickly. After the reward stabilizes, the gradient norm decreases. With it the sharpness of the parameters and the BT loss also decrease. This demonstrates that the gradient norm is tied to both the sharpness and accuracy of the PR. Continuing training after the reset, we now start out with a more accurate RM. This enables training in a regime with a good PR, leading to a better final policy. 

**Reference Resets Outperform Standard KL** We run an extensive comparison across both families and two sizes per model on standard baselines. The initial models are SFTtrained on TLDR data. From there we compare our method to DPO (Rafailov et al., 2023), DPO with reference reset, also known as Trust Region DPO (TR-DPO) (Gorbatovski et al., 2025), and standard GRPO with a fixed KL penalty. Our results are shown in Table 1. GRPO + Reference Resets strongly outperforms all other methods for all but one 

5 

**Gradient Regularization Prevents Reward Hacking** 

**==> picture [476 x 127] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 . 66<br>3 2 . 1<br>1 − 8<br>0 . 64<br>2 0 . 8 2 − 10 β = 0 . 03<br>0 . 62 β = 0 . 04<br>β = 0 . 05<br>1 1 . 9 − 12 β = 0 . 06<br>0 . 6 0 . 6 β = 0 . 07<br>β = 0 . 08<br>1 . 8 − 14 Resets<br>0<br>0 1000 2000 0 1000 2000 0 1000 2000 0 50 100<br>Step Step Step D KL( πθ||π 1)<br>Sharpness BT Loss<br>Training Reward Gradient Norm Gold Model Score<br>**----- End of picture text -----**<br>


_Figure 3._ **When gradient norm decreases in a reset iteration, so do sharpness, and BT Loss** _L_ RM **.** Evolution of reward, sharpness and BT loss during training on TL;DR with Pythia 1B using GRPO+reference resets, resets shown as grey dashed lines. After initially spiking in an iteration, gradient norm decreases along with the sharpness of the parameters and the BT-loss under the current policy. We show moving averages over 30 steps. 

_Figure 4._ **Reference Resets outperform all possible weights** _β_ **of KL penalty.** Oracle evaluation (Gold Model Score) vs KL from initial model for Pythia 1B on the TL;DR test set. 

_Table 1._ Win rate vs reference response on the TL;DR summarization task, as judged by the gold RM. 

|Model Family<br>Size|<br>Pythia<br>1B<br>2.8B|
|---|---|
|SFT<br>DPO<br>TR-DPO|17.8%<br>25.6%<br>45.9%<br>68.0%<br>45.5%<br>66.6%|
|GRPO<br>GRPO + Ref.|62.2%<br>68.5%<br> Reset<br>**78.1%**<br>**76.4%**|



setup. Notably, a Pythia 1B model trained with Reference Resets performs better than a Pythia 2.8B model trained with standard GRPO, even with a finely-tuned _β_ , as shown in Appendix C. In our experiments, TR-DPO’s Reference Resets do not seem to afford the same performance improvements. We speculate that this is because DPO does not use an RM, thus there can be no sharp action-space PR maxima which GR would help to avoid. 

## **5. Explicit Gradient Regularization** 

Reference Resets are an indirect way of regularizing the gradient norm, require many more gradient steps, and do not provide a direct, controllable way to trade-off reward maximization, adherence to the initial policy, and gradient regularization. We therefore propose a novel application of explicit GR methods to RLHF and RLVR, specifically finite-difference GR (Karakida et al., 2023). To improve training stability, we implement parameter perturbations only on the transformer blocks, leaving the embedding layer and output head untouched, and clip the intermediate gradients. To make GR training efficient, we reuse the actions _a ∼ πϕ_ ( _a|s_ ) to calculate both the gradients _∇J_ ( _ϕ, θ_ ) and _∇J_ ( _ϕ_ + _ε∇J_ ( _ϕ, θ_ ) _, θ_ ). In principle, this would require new actions _a ∼ πϕ_ + _ε∇J_ ( _ϕ,θ_ )( _a|s_ ) or correction by importance sampling. However, we empirically found this to be unnecessary and reusing actions reduces computation overhead.[1] PyTorch-style pseudocode, implementation details and experimental details are shown in Appendix C. 

## **5.1. GR Mitigates Hacking Reward Models in RLHF** 

**Is a Scheduled or Weaker** _β_ **a Sufficient Alternative?** Originally, Reference Resets were proposed (Liu et al., 2025a) not to improve RM accuracy but to prevent the KL term from dominating the reward sum. If this was their main mechanism, decreasing the strength of the KL penalty _β_ should have a similar effect. Further, an analysis of the optimal policy under iterated KL constrained optimization (see Appendix E) shows that the optimal policy for _i_ reset iterations is equivalent to the optimal policy with a KL constraint to _π_[1] with a lower penalty strength _β[′]_ = _β/i_ . However, Figure 4 shows that decreasing _β_ is not able to match Reference Resets. This demonstrates the necessity for our novel insight that Reference Resets change the optimization dynamics via implicit gradient regularization during training and its effects on the PR accuracy. 

We first investigate whether GR can fully replace the standard KL penalty in RLHF. Scaling up from TL;DR, we run experiments on the AlpacaFarm dataset (Dubois et al., 2023), with preference feedback from GPT 4.1-Nano. We again evaluate with winrate: generating completions on the test set and judging them against reference completions with GPT4.1-Nano. We train models from the Qwen 2.5 family, running for 1000 gradient steps in the 1.5B experiments and 500 gradient steps in the 0.5B and 3B experiments and earlystop based on the training winrate. We evaluate and ablate GR in four settings: 1) no KL penalty, no GR 2) KL penalty, 

> 1In our MATH reasoning experiments on 8 GH200 GPUs, generating the actions took on average 7.4s, while the policy update took 60ms without GR and 150ms with GR. 

6 

**Gradient Regularization Prevents Reward Hacking** 

_Table 2._ Win rate vs reference response on AlpacaFarm dataset, Qwen 2.5, judged by GPT4.1-Nano with early stopping. The 1.5B experiment is run for three different SFT policies and RMs. 

|<br><br><br>|<br><br><br>|
|---|---|
|Model Size<br>0.5B<br>1.5B<br>3B||
|SFT Model<br>12.8%<br>20.6%<br>26.3%||
|No Reg<br>12.8%<br>21.7%<br>44.2%<br>KL Reg<br>16.9%<br>27.6%<br>52.8%<br>Reference Resets<br>17.4%<br>27.1%<br>49.2%<br>GR<br>**18.5%**<br>**29.2%**<br>**59.2%**||
|20<br>30<br>40<br>Winrate (%)|0_._6<br>0_._65<br>0_._7<br>RM Accuracy<br>SFT<br>No Reg<br>KL Reg<br>Reference Reset<br>GR|



_Figure 5._ **Explicit GR performs well even with inaccurate RMs.** RM accuracy on SFT data vs GPT 4.1 Accuracy for different SFT and RM models, corresponding to different random seeds for full RLHF pipeline. The x-axis scale is nonlinear. 

3) Reference Resets + KL, 4) GR without KL. We do a gridsearch to find the optimal strength for each regularization method for each model size and show results in Table 2. In most cases using RL without regularization only yields a modest improvement over the initial SFT policy. RL with a KL penalty works decently well though Reference Resets is sometimes better. But explicit GR consistently performs best, demonstrating that it can replace the KL penalty and improve overall performance. 

**Explicit GR is robust to RM accuracy** RLHF performance can vary heavily depending on the accuracy of the RM and performance of the initial policy (Huang et al., 2024). To demonstrate robustness, we rerun our whole pipeline two more times for the 1.5B model: initial SFT, dataset sampling, RM training, GRPO with hyper-parameter tuning. In Figure 5 we show the winrate of each trained model against the accuracy of the RM with which it trained. We observe that GR performs significantly better than Reference Resets or a KL penalty when the RM is weaker, demonstrating better robustness. Reference Resets do perform slightly better than GR with the strongest RM, where reward-hacking is less prevalent. 

We also find that GR is robust to choices of hyperparameters _γ_ and _ε_ , and show a learning rate sweep in Appendix D.1. Finally, we discover that strong GR can even compensate in robustness for a weak RM by training in a 

_Table 3._ Test accuracies on GSM8K after with GRPO and different regularization methods, with LLM judge or rule-based reward. 

|Feedback-Type<br>Qwen 2.5 Size|Feedback-Type<br>Qwen 2.5 Size|Feedback-Type<br>Qwen 2.5 Size|Feedback-Type<br>Qwen 2.5 Size|Feedback-Type<br>Qwen 2.5 Size|Rule-Based<br>0.5B<br>1.5B|LLM Judge<br>0.5B<br>1.5B|LLM Judge<br>0.5B<br>1.5B|LLM Judge<br>0.5B<br>1.5B|
|---|---|---|---|---|---|---|---|---|
|Base model|||||3.0%<br>37.5%|3.0%<br>37.5%|||
|No Reg<br>KL Reg<br>GR|||||46.7%<br>72.9%<br>44.3%<br>72.4%<br>**50.9**%<br>**75.7**%|2.0%<br>1.9%<br>26.4%<br>55.5%<br>**42.8**%<br>**67.8**%|||
|0_._2<br>0_._4<br>Accuracy||||||0<br>500<br>1000<br>Gradient Step|||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
|||||No Reg<br>GR|||||
||||<br>||||||
||||||||||
|||||||0<br>500<br>10<br>Gradient Step|||



_Figure 6._ **GR prevents overly focusing on formatting reward.** Qwen2.5-0.5B on GSM8K, test set accuracy (left) and formatting reward (right), the dashed line shows the optimal formatting reward. Without regularization, the policy focuses overly on the formatting reward, resulting in worse accuracy. 

regime where the RM is more accurate than on its training distribution, see Appendix D.2. 

## **5.2. GR Prevents Focus on Easy Rule-Based Rewards** 

Recent work in RLVR has generally removed the KL penalty to allow for a stronger deviation from the base model (GLM4.5 Team et al., 2025; Olmo Team et al., 2025), though others have kept it for training stability (Kimi Team et al., 2025). As GR does not constrain the divergence from the base model, but may provide the desired training stability, we investigate whether it can be used in RLVR to enable more flexibility while preventing reward hacking. 

We perform experiments with Qwen 2.5 0.5B-Instruct and 1.5B-Instruct on GSM8K (Cobbe et al., 2021) with the standard combination of formatting and correctness reward, see full details in Appendix C. We indeed observe more stable training and improved final accuracy, as shown in Figure 6 (left) and Table 3. Notably, improved performance comes at the expense of a slightly worse adherence to the formatting reward. In the presence of both rewards, we can see the excessive focus on the easier formatting reward as a kind of reward hacking, even when neither reward is hackable on its own. This demonstrates how GR can be effective in situations with a combination of rule-based rewards. 

Next, we show that a form of reward hacking is also possible even within a single reward and how GR can mitigate it. We 

7 

**Gradient Regularization Prevents Reward Hacking** 

_Table 4._ **GR prevents focus on easy questions.** Accuracy on MATH with rule-based reward, by difficulty of the question category, grouped by base-model accuracy. Colored depending on increase or decrease after Step 250. Without regularization, after initial improvement, the policy improves on the easy questions but worsens on the hard questions. 

|Step<br>0<br>250<br>500<br>750<br>1000<br>Init. Acc.|Step<br>0<br>250<br>500<br>750<br>1000<br>Init. Acc.|
|---|---|
|GR<br>00-20%<br>11.7<br>15.4<br>15.2<br>16.2<br>16.2<br>20-40%<br>31.9<br>39.8<br>40.2<br>41.1<br>41.8<br>40-60%<br>49.1<br>57.9<br>58.4<br>59.4<br>58.7<br>60-80%<br>68.0<br>75.7<br>76.4<br>76.7<br>76.5<br>>80%<br>83.0<br>87.9<br>89.0<br>89.6<br>89.2||
|No Reg<br>00-20%<br>11.7<br>15.2<br>14.9<br>13.9<br>14.3<br>20-40%<br>31.9<br>39.0<br>38.5<br>36.7<br>37.6<br>40-60%<br>49.1<br>57.7<br>58.3<br>55.8<br>56.2<br>60-80%<br>68.0<br>76.2<br>76.0<br>74.3<br>73.9<br>>80%<br>83.0<br>87.0<br>88.0<br>87.2<br>87.9||
|0_._7<br>0_._8<br>0_._9<br>LLM Judge Score|0<br>500<br>1000<br>Gradient Steps<br>0<br>500<br>1000<br>0<br>0_._2<br>0_._4<br>Gradient Steps<br>True Accuracy<br>No Reg<br>GR<br>KL|



_Figure 7._ **GR prevents reward hacking with LLM-as-a-Judge** Results when training Qwen2.5-0.5B-Inst. on GSM8K with Qwen2.5 1.5B-Inst. as judge. Left: LLM-Judge and rule-based accuracy over time, showing reward hacking without regularization. 

train a Qwen-2.5-1.5B-Instruct model with a rule-based reward on MATH (Hendrycks et al., 2021). The base model achieves 46.3% pass@1 accuracy and GRPO+GR (57.6%) clearly outperforms standard GRPO (54.8%). In order to discern reward hacking, we investigate the accuracy more closely. We divide performance into quintiles of difficulty so that test-set questions are split based on the initial accuracy of their category and level-labels, as provided by the dataset As shown in Table 4, without regularization the performance on the easiest quintile of questions continues to improve past 250 steps. But after the first 250 steps, performance on harder questions actually degrades. This demonstrates how RL can focus on learning only the easiest questions in order to hack even a rule-based reward. In contrast, training with GR more evenly improves the accuracies across difficulties, avoiding the focus on the easier questions. 

## **5.3. GR Mitigates Hacking LLM-as-a-Judge in RLVR** 

Finally, we investigate RL with LLM-as-a-Judge. As LLMas-a-Judge can be susceptible to adversarial attacks (Zhao et al., 2025), we presume it can also be reward-hacked as a 

**==> picture [232 x 111] intentionally omitted <==**

**----- Start of picture text -----**<br>
No Regularization GR<br>0.4<br>0.3 1.0<br>0.2<br>0.1<br>0.0<br>0 500 1000 0 500 1000<br>Gradient Step Gradient Step<br>Accuracy<br>Gradient Norm<br>**----- End of picture text -----**<br>


_Figure 8._ **Gradient norm increases as reward hacking occurs.** True accuracy and gradient norm when training Qwen2.5 0.5BInstruct on GSM8K with Qwen2.5 1.5B-Instruct LLM judge. 

PR. To investigate, we re-run the previous GSM8k experiments but replace the rule-based correctness reward with an LLM judge based on Qwen2.5 1.5B-Instruct (Qwen et al., 2025). The judge receives the problem description, true answer, model response, model reasoning, and is instructed to output a score for correctness (1 to 5) which acts as the reward. As shown in Figure 7, without regularization the model quickly starts to “hack” the judge LLM. LLM judge score goes sharply up while pass@1 accuracy on the test set peaks quite early. Empirically, we observed the model outputting excessive brackets and new HTML tags to fool the judge. In contrast, both GR and KL show much more reasonable train rewards and prevent excessive reward hacking. But as shown in Table 3, GR results in better final performance. Confirming results in Section 4, we also see that, without regularization, an increase in gradient norm occurs when reward hacking begins, shown in Figure 8 (left). In Appendix D.3 we compare different judge models, showing that the additional cost of GR can be amortized by allowing the usage of smaller judge models. 

## **6. Conclusion** 

We have investigated the problem of RL with proxy rewards (PR) and proposed a novel perspective: learning a policy in a regime where the PR is accurate. For this purpose, we have derived a theoretical connection between the flatness of an optimum and the Bradley-Terry loss of the PR at this optimum. By regularizing the gradient norm during training, we can bias the RL updates towards such a flat optimum. We first validated our theoretical analysis by using implicit gradient regularization (GR) via Reference Resets, showing they improve upon a KL penalty. We then proposed to use explicit GR based on an efficient implementation of a finite-difference estimate. Explicit GR allows us to mitigate reward model hacking of RMs in RLHF, reduce focus on easy rule-based rewards in RLVR, and alleviate format hacking with LLM-as-a-judge. We believe that GR is a promising candidate to completely replace KL penalties and improve training runs that currently eschew regularization. 

8 

**Gradient Regularization Prevents Reward Hacking** 

## **Acknowledgements** 

We would like to thank Soichiro Nishimori and Thanawat Lodkaew for helpful discussions. 

## **Impact Statement** 

We propose a method to prevent reward hacking in RL post-training of LLMs. We believe that preventing reward hacking is likely to have a beneficial impact in general. However, our theory only considers certain specific kinds of reward hacking, thus there is a risk that by overly relying on our method, users may miss other kinds of reward hacking, which should be monitored independently. 

## **References** 

- Ackermann, J., Ishida, T., and Sugiyama, M. Off-Policy Corrected Reward Modeling for Reinforcement Learning from Human Feedback. In _COLM_ , 2025. URL https: //openreview.net/forum?id=0zxugBcgF5. 

- Ahmadian, A., Cremer, C., Gallé, M., Fadaee, M., Kreutzer, J., Pietquin, O., Üstün, A., and Hooker, S. Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs. In _ACL_ , 2024. URL https://aclanthology.org/2024. acl-long.662.pdf. 

- Bahri, D., Mobahi, H., and Tay, Y. Sharpness-Aware Minimization Improves Language Model Generalization. In _ACL_ , pp. 7360–7371, Dublin, Ireland, 2022. URL https://aclanthology.org/2022. acl-long.508/. 

- Barrett, D. G. T. and Dherin, B. Implicit Gradient Regularization. In _ICLR_ , 2021. URL http://arxiv.org/ abs/2009.11162. 

- Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., and Wal, O. v. d. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. In _ICML_ , 2023. URL http://arxiv.org/abs/ 2304.01373. 

- Bradley, R. A. and Terry, M. E. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. _Biometrika_ , 39(3/4):324–345, 1952. URL https:// www.jstor.org/stable/2334029. 

- Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. In _NeurIPS 2017_ , 2017. URL http: //arxiv.org/abs/1706.03741. 

- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training Verifiers to Solve Math Word Problems, 2021. URL http://arxiv.org/abs/2110.14168. 

- DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. Technical report, 2025. URL https://arxiv.org/abs/ 2501.12948. 

- Dubois, Y., Li, X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., Guestrin, C., Liang, P., and Hashimoto, T. B. AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback. In _NeurIPS_ . arXiv, 2023. URL http: //arxiv.org/abs/2305.14387. 

- Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. KTO: Model Alignment as Prospect Theoretic Optimization. In _ICML_ , 2024. URL http://arxiv. org/abs/2402.01306. 

- Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. Sharpness-Aware Minimization for Efficiently Improving Generalization. In _ICLR_ , 2021. URL http://arxiv. org/abs/2010.01412. 

- Gao, L., Schulman, J., and Hilton, J. Scaling Laws for Reward Model Overoptimization. In _ICML_ , 2023. URL https://proceedings.mlr.press/ v202/gao23h/gao23h.pdf. 

- GLM-4.5 Team, Zeng, A., Lv, X., Zheng, Q., Hou, Z., Chen, B., Xie, C., Wang, C., Yin, D., Zeng, H., Zhang, J., Wang, K., Zhong, L., Liu, M., Lu, R., Cao, S., Zhang, X., Huang, X., Wei, Y., Cheng, Y., An, Y., Niu, Y., Wen, Y., Bai, Y., Du, Z., Wang, Z., Zhu, Z., Zhang, B., Wen, B., Wu, B., Xu, B., Huang, C., Zhao, C., Cai, C., Yu, C., Li, C., Ge, C., Huang, C., Zhang, C., Xu, C., Zhu, C., Li, C., Yin, C., Lin, D., Yang, D., Jiang, D., Ai, D., Zhu, E., Wang, F., Pan, G., Wang, G., Sun, H., Li, H., Li, H., Hu, H., Zhang, H., Peng, H., Tai, H., Zhang, H., Wang, H., Yang, H., Liu, H., Zhao, H., Liu, H., Yan, H., Liu, H., Chen, H., Li, J., Zhao, J., Ren, J., Jiao, J., Zhao, J., Yan, J., Wang, J., Gui, J., Zhao, J., Liu, J., Li, J., Li, J., Lu, J., Wang, J., Yuan, J., Li, J., Du, J., Du, J., Liu, J., Zhi, J., Gao, J., Wang, K., Yang, L., Xu, L., Fan, L., Wu, L., Ding, L., Wang, L., Zhang, M., Li, M., Xu, M., Zhao, M., Zhai, M., Du, P., Dong, Q., Lei, S., Tu, S., Yang, S., Lu, S., Li, S., Li, S., Shuang-Li, Yang, S., Yi, S., Yu, T., Tian, W., Wang, W., Yu, W., Tam, W. L., Liang, W., Liu, W., Wang, X., Jia, X., Gu, X., Ling, X., Wang, X., Fan, X., Pan, X., Zhang, X., Zhang, X., Fu, X., Zhang, X., Xu, Y., Wu, Y., Lu, Y., Wang, Y., Zhou, Y., Pan, Y., Zhang, Y., Wang, Y., Li, Y., Su, Y., Geng, Y., Zhu, Y., Yang, Y., Li, Y., Wu, Y., Li, Y., Liu, Y., Wang, Y., Li, Y., Zhang, Y., Liu, Z., 

9 

**Gradient Regularization Prevents Reward Hacking** 

- Yang, Z., Zhou, Z., Qiao, Z., Feng, Z., Liu, Z., Zhang, Z., Wang, Z., Yao, Z., Wang, Z., Liu, Z., Chai, Z., Li, Z., Zhao, Z., Chen, W., Zhai, J., Xu, B., Huang, M., Wang, H., Li, J., Dong, Y., and Tang, J. GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models, 2025. URL http://arxiv.org/abs/2508.06471. 

- Gorbatovski, A., Shaposhnikov, B., Malakhov, A., Surnachev, N., Aksenov, Y., Maksimov, I., Balagansky, N., and Gavrilov, D. Learn Your Reference Model for Real Good Alignment. In _ICLR_ , 2025. URL http: //arxiv.org/abs/2404.09656. 

- Havrilla, A., Du, Y., Raparthy, S. C., Nalmpantis, C., Dwivedi-Yu, J., Zhuravinskyi, M., Hambro, E., Sukhbaatar, S., and Raileanu, R. Teaching Large Language Models to Reason with Reinforcement Learning. In _ICML 2024 Workshop AI4Math_ . arXiv, 2024. URL http://arxiv.org/abs/2403.04642. 

- Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring Mathematical Problem Solving With the MATH Dataset. In _NeurIPS_ , 2021. URL http://arxiv.org/abs/ 2103.03874. 

- Hochreiter, S. and Schmidhuber, J. Flat Minima. _Neural Computation_ , 9(1):1–42, 1997. URL https://doi. org/10.1162/neco.1997.9.1.1. 

- Huang, S., Noukhovitch, M., Hosseini, A., Rasul, K., Wang, W., and Tunstall, L. The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization. In _COLM_ , 2024. URL http://arxiv.org/ abs/2403.17031. 

- Hugging Face. Open r1: A fully open reproduction of deepseek-r1, 2025. URL https://github.com/ huggingface/open-r1. 

- Karakida, R., Takase, T., Hayase, T., and Osawa, K. Understanding Gradient Regularization in Deep Learning: Efficient Finite-Difference Computation and Implicit Bias. In _ICML_ , pp. 15809–15827, 2023. URL https://proceedings.mlr.press/ v202/karakida23a.html. 

- Kimi Team, Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., Chen, Z., Cui, J., Ding, H., Dong, M., Du, A., Du, C., Du, D., Du, Y., Fan, Y., Feng, Y., Fu, K., Gao, B., Gao, H., Gao, P., Gao, T., Gu, X., Guan, L., Guo, H., Guo, J., Hu, H., Hao, X., He, T., He, W., He, W., Hong, C., Hu, Y., Hu, Z., Huang, W., Huang, Z., Huang, Z., Jiang, T., Jiang, Z., Jin, X., Kang, Y., Lai, G., Li, C., Li, F., Li, H., Li, M., Li, W., Li, Y., Li, Y., Li, Z., Li, Z., Lin, H., Lin, X., Lin, Z., Liu, C., Liu, C., Liu, H., Liu, J., Liu, J., Liu, L., Liu, S., Liu, T. Y., Liu, T., 

Liu, W., Liu, Y., Liu, Y., Liu, Y., Liu, Y., Liu, Z., Lu, E., Lu, L., Ma, S., Ma, X., Ma, Y., Mao, S., Mei, J., Men, X., Miao, Y., Pan, S., Peng, Y., Qin, R., Qu, B., Shang, Z., Shi, L., Shi, S., Song, F., Su, J., Su, Z., Sun, X., Sung, F., Tang, H., Tao, J., Teng, Q., Wang, C., Wang, D., Wang, F., Wang, H., Wang, J., Wang, J., Wang, J., Wang, S., Wang, S., Wang, Y., Wang, Y., Wang, Y., Wang, Y., Wang, Y., Wang, Z., Wang, Z., Wang, Z., Wei, C., Wei, Q., Wu, W., Wu, X., Wu, Y., Xiao, C., Xie, X., Xiong, W., Xu, B., Xu, J., Xu, J., Xu, L. H., Xu, L., Xu, S., Xu, W., Xu, X., Xu, Y., Xu, Z., Yan, J., Yan, Y., Yang, X., Yang, Y., Yang, Z., Yang, Z., Yang, Z., Yao, H., Yao, X., Ye, W., Ye, Z., Yin, B., Yu, L., Yuan, E., Yuan, H., Yuan, M., Zhan, H., Zhang, D., Zhang, H., Zhang, W., Zhang, X., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Z., Zhao, H., Zhao, Y., Zheng, H., Zheng, S., Zhou, J., Zhou, X., Zhou, Z., Zhu, Z., Zhuang, W., and Zu, X. Kimi K2: Open Agentic Intelligence, 2025. URL https://arxiv.org/abs/2507.20534. 

- Kingma, D. P. and Ba, J. Adam: A Method for Stochastic Optimization. In _ICLR_ , 2015. URL http://arxiv. org/abs/1412.6980. 

- Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, S., Gu, Y., Malik, S., Graf, V., Hwang, J. D., Yang, J., Bras, R. L., Tafjord, O., Wilhelm, C., Soldaini, L., Smith, N. A., Wang, Y., Dasigi, P., and Hajishirzi, H. Tulu 3: Pushing Frontiers in Open Language Model Post-Training, 2024. URL http://arxiv.org/abs/2411.15124. 

- Lee, H., Cho, H., Kim, H., Gwak, D., Kim, J., Choo, J., Yun, S.-Y., and Yun, C. PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning. In _NeurIPS_ , 2023. URL http://arxiv.org/abs/ 2306.10711. 

- Lee, H. K. and Yoon, S. W. Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning. In _ICLR_ , 2025. URL https://openreview.net/forum? id=4OaO3GjP7k. 

- Liu, C. Y., Zeng, L., Liu, J., Yan, R., He, J., Wang, C., Yan, S., Liu, Y., and Zhou, Y. Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs, 2024. URL http://arxiv.org/abs/2410.18451. 

- Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., and Dong, Y. ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models, 2025a. URL http://arxiv.org/ abs/2505.24864. 

- Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. Understanding R1-Zero-Like Training: 

10 

**Gradient Regularization Prevents Reward Hacking** 

- A Critical Perspective, 2025b. URL http://arxiv. org/abs/2503.20783. 

- Loshchilov, I. and Hutter, F. Decoupled Weight Decay Regularization. In _ICLR_ , 2019. URL http://arxiv. org/abs/1711.05101. 

- Nikishin, E., Schwarzer, M., D’Oro, P., Bacon, P.L., and Courville, A. The Primacy Bias in Deep Reinforcement Learning. In _ICML_ , June 2022. URL https://proceedings.mlr.press/ v162/nikishin22a.html. 

- Noukhovitch, M., Lavoie, S., Strub, F., and Courville, A. Language Model Alignment with Elastic Reset. In _NeurIPS_ , 2023. URL https://openreview.net/ forum?id=6lgugutkin. 

- Olmo Team, Ettinger, A., Bertsch, A., Kuehl, B., Graham, D., Heineman, D., Groeneveld, D., Brahman, F., Timbers, F., Ivison, H., Morrison, J., Poznanski, J., Lo, K., Soldaini, L., Jordan, M., Chen, M., Noukhovitch, M., Lambert, N., Walsh, P., Dasigi, P., Berry, R., Malik, S., Shah, S., Geng, S., Arora, S., Gupta, S., Anderson, T., Xiao, T., Murray, T., Romero, T., Graf, V., Asai, A., Bhagia, A., Wettig, A., Liu, A., Rangapur, A., Anastasiades, C., Huang, C., Schwenk, D., Trivedi, H., Magnusson, I., Lochner, J., Liu, J., Miranda, L., Sap, M., Morgan, M., Schmitz, M., Guerquin, M., Wilson, M., Huff, R., Bras, R. L., Xin, R., Shao, R., Skjonsberg, S., Shen, S. Z., Li, S. S., Wilde, T., Pyatkin, V., Merrill, W., Chang, Y., Gu, Y., Zeng, Z., Sabharwal, A., Zettlemoyer, L., Koh, P. W., Farhadi, A., Smith, N. A., and Hajishirzi, H. Olmo 3, 2025. URL https://arxiv.org/abs/2512.13961. 

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., and Lowe, R. Training language models to follow instructions with human feedback. In _NeurIPS_ , 2022. URL http://arxiv.org/abs/2203.02155. 

- Qwen, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 Technical Report, 2025. URL http://arxiv.org/abs/2412.15115. 

- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In 

- _NeurIPS_ , 2023. URL https://arxiv.org/abs/ 2305.18290v2. 

- Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. In _SC ’20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_ , 2020. URL http://arxiv.org/abs/1910.02054. 

- Razin, N., Malladi, S., Bhaskar, A., Chen, D., Arora, S., and Hanin, B. Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization. In _ICLR_ , 2025. URL http://arxiv.org/abs/2410. 08847. 

- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, 2024. URL http://arxiv.org/abs/2402.03300. 

- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. Learning to summarize from human feedback. In _NeurIPS_ , 2020. URL http://arxiv.org/abs/ 2009.01325. 

- Tang, Y., Guo, D. Z., Zheng, Z., Calandriello, D., Cao, Y., Tarassov, E., Munos, R., Pires, B. Á., Valko, M., Cheng, Y., and Dabney, W. Understanding the performance gap between online and offline alignment algorithms, 2024. URL http://arxiv.org/abs/2405.08448. 

- von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., Huang, S., Rasul, K., and Gallouédec, Q. Trl: Transformer reinforcement learning. https://github.com/huggingface/trl, 2020. 

- Wei, C., Yu, J., He, Y. T., Dong, H., Shu, Y., and Yu, F. ReDit: Reward Dithering for Improved LLM Policy Optimization. In _NeurIPS_ , 2025. URL http: //arxiv.org/abs/2506.18631. 

- Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_ , 8(3-4):229–256, 1992. URL http://link. springer.com/10.1007/BF00992696. 

- Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, 

- H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, 

- J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, 

11 

**Gradient Regularization Prevents Reward Hacking** 

- S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, 

- X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 Technical Report, 2025. URL http: //arxiv.org/abs/2505.09388. 

- Zhang, Z., Luo, R., Su, Q., and Sun, X. GA-SAM: GradientStrength based Adaptive Sharpness-Aware Minimization for Improved Generalization. In Goldberg, Y., Kozareva, Z., and Zhang, Y. (eds.), _EMNLP_ , Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. URL https://aclanthology.org/ 2022.emnlp-main.257/. 

- Zhao, Y., Zhang, H., and Hu, X. Penalizing Gradient Norm for Efficiently Improving Generalization in Deep Learning. In _ICML_ , 2022. URL https://proceedings. mlr.press/v162/zhao22i.html. 

- Zhao, Y., Zhang, H., and Hu, X. When Will Gradient Regularization Be Harmful? In _ICML_ , 2024. URL http://arxiv.org/abs/2406.09723. 

- Zhao, Y., Liu, H., Yu, D., Kung, S., Chen, M., Mi, H., and Yu, D. One Token to Fool LLM-as-a-Judge, September 2025. URL http://arxiv.org/abs/2507. 08794. 

- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. Judging LLM-as-aJudge with MT-Bench and Chatbot Arena. In _NeurIPS Datasets & Benchmarks_ , 2023. URL http://arxiv. org/abs/2306.05685. 

12 

**Gradient Regularization Prevents Reward Hacking** 

## **A. Extended related work** 

**Reference Resets** We here provide an overview of alternative explanations for reference-resets and related methods. Noukhovitch et al. _ϕ_ ¯ during the updates. (2023After each) proposed Elastic Resets, a method which maintains an exponential moving average of the weights _R_ steps, the current policy parameters are replaced with the EMA weights _ϕ_ ¯ and the EMA weights are reinitialized to the initial policy _ϕ_[1] . Next, (Gorbatovski et al., 2025) proposed Trust-Region Direct Preference Optimization (TR-DPO), which proposes a method very similar to reference resets, applied to DPO (Rafailov et al., 2023). Similar to Reference Resets, the reference policy is set to the current policy after each _R_ updates. While DPO is meant to increase the probability of preferred responses _π_ ( _a_ w _|s_ ) while decreasing the probability of un-preferred responses _π_ ( _a_ l _|s_ ), in practice it often decreases both _π_ ( _a_ w _|s_ ) and _π_ ( _a_ l _|s_ ), degrading performance. This issue is known as likelihood displacement (Razin et al., 2025). Gorbatovski et al. (2025) motivate the mechanism of TR-DPO by preventing likelihood displacement, however likelihood displacement does not occur in on-policy alignment methods such as GRPO, which we investigate in this work. This mechanism can thus not explain why Reference Resets improve over standard GRPO. 

Liu et al. (2025a) motivate the need for Reference Resets in reasoning training due to the magnitude of the KL penalty overwhelming the magnitude of the reward after a certain number of training steps. However, if this was the reason, we could simply use a smaller KL penalty _β_ from the beginning, which we show in Figure 15 does not work. Another reasonable approach following the reasoning of (Liu et al., 2025a) would be to simply decrease _β_ when the reward stagnates, but we also show in the experiments (Figure 14) that this does not work as well as Reference Resets. Instead, we provide a theoretical analysis explaining its success and provide experiments validating it specifically in the RLHF setting. Ackermann et al. (2025) contemporarily to Liu et al. (2025a) proposed reference resets as an ablation called “PPO + New KL”. 

**Replacing the KL penalty in Post-training** The most common way to prevent reward hacking with proxy rewards is a KL penalty, introduced in RLHF by Stiennon et al. (2020) and RLVR by Havrilla et al. (2024). Some recent reports about large model training, for example GLM4.5 (GLM-4.5 Team et al., 2025) and Olmo3 (Olmo Team et al., 2025) remove the KL penalty, while others such as Kimi K2 (Kimi Team et al., 2025) still use it. Our experiments show that both in RLHF and math experiments using gradient regularization performs better than either the KL penalty or removing the KL penalty. Direct alignment methods such as Direct Preference Optimization (DPO) (Rafailov et al., 2023), Kahneman-Tversky Optimization (KTO) (Ethayarajh et al., 2024), do not use an explicit KL penalty during training, however they indirectly optimize the KL-constrained RL objective. Instead of just a KL penalty (Ouyang et al., 2022) uses _PPO-ptx_ which adds an additional behavior cloning term to the loss, on data sampled from a pretraining dataset, to prevent regressions on standard NLP benchmarks during RLHF training. 

**Gradient Regularization** While (Karakida et al., 2023) performs experiments with convolutional neural networks, (Zhao et al., 2022; 2024) further apply GR to vision transformer models. Sharpness-aware minimization (SAM) (Foret et al., 2021) has been shown to correspond to gradient regularization with a specific choice of hyper-parameters (Karakida et al., 2023). (Bahri et al., 2022; Zhang et al., 2022) apply SAM to transformer pre-training in order to improve generalization. SAM has also been used in RL, particularly by Lee et al. (2023) to improve sample efficiency when training a policy for Atari games, and by (Lee & Yoon, 2025) to obtain robust policies in continuous control tasks. Our theoretical analysis in part uses the argument provided (Lee & Yoon, 2025), relating parameter flatness to action robustness. Our work is, to our knowledge, the first one to investigate the relation of GR to the accuracy of proxy rewards, as well as the first work to use GR in RLHF/RLVR post-training of LMs. 

## **B. Proof(s)** 

Our argument can be summarized as follows: Lee & Yoon (2025) showed that flatness in parameter space is related to flatness in action space, we slightly extend the argument to a maximum error of _L_[�] in Proposition B.5. In our case this controls the expected proxy reward inside, when assuming a Gaussian policy with fixed covariance and full row-rank Jacobian (Assumption B.2). Under the assumption of _β_ smoothness of the expected proxy reward (Assumption B.3), this provides a bound on the gradient norm of the expected proxy reward (Lemma B.6). We then relate the gradient norm of the expected proxy reward to the point-wise norm of the action-gradient in Lemma B.7. Then, as in an area with a bounded gradient norm Lipschitz continuity is guaranteed, in Lemma B.8 we show that overly sharp action pairs can only occur if at least one of the actions is outside of a certain radius from the maximum. Putting it all together yields Proposition 3.3. Finally, we show that, under a Lipschitz continuous true reward (Assumption B.4), sharp minima incur an excess BT error. 

13 

**Gradient Regularization Prevents Reward Hacking** 

## **B.1. Flatness in Parameter Space implies** ( _δ, K, ρ_ ) **-pairwise robustness** 

We will first show that a _E − L_[�] flat reward implies a _D − L_[�] robust policy, closely based on the proof proposed by Lee & Yoon (2025). We also need to define a _D − L_[�] action robust policy: 

**Definition B.1** ( _D − L_[�] action robust policy, slightly modified from Lee & Yoon (2025)) **.** For a reward function _R_ ( _s, a_ ) and policy _πϕ_ ( _a|s_ ), parameterized by _ϕ_ , a maximum _ϕ[∗]_ is _D_ - _L_[�] action robust if the following holds: 

**==> picture [420 x 19] intentionally omitted <==**

We will also need the following assumptions during this section: 

**Assumption B.2** (Gaussian policy with fixed covariance and full row-rank Jacobian) **.** For each _s_ , _πϕ∗_ ( _· | s_ ) = _N_ ( _µϕ∗_ ( _s_ ) _,_ Σ) with a fixed positive definite covariance Σ. The Jacobian matrix J( _ϕ[∗]_ ) of the mean action _µϕ_ ( _s_ ) w.r.t. _ϕ_ , evaluated at _ϕ[∗]_ , has full row rank. 

**Assumption B.3** (Action-smooth proxy reward) **.** For all _s ∈S_ , the PR _R_[�] ( _s, a_ ) is _β_ -smooth in _a_ , i.e., _∥∇aR_[�] ( _s, a_ 1) _− ∇aR_[�] ( _s, a_ 2) _∥≤ β∥a_ 1 _− a_ 2 _∥_ for all _a_ 1 _, a_ 2 _∈A_ . 

**Assumption B.4** (Lipschitz-continuous true reward) **.** For all _s ∈S_ , the true reward _R[∗]_ ( _s, a_ ) is Lipschitz in _a_ , i.e., _|R[∗]_ ( _s, a_ 1) _− R[∗]_ ( _s, a_ 2) _| ≤ L∥a_ 1 _− a_ 2 _∥_ for all _a_ 1 _, a_ 2. 

We now can show that an _E − L_[�] flat return implies _D − L_[�] robust policy: 

**Proposition B.5** ( _E − L_[�] flat return implies _D − L_[�] robust policy, slightly modified from Lee & Yoon (2025)) **.** _If ϕ[∗] is an E − L_[�] _flat return maximum of a policy under assumption B.2, then the policy ϕ[∗] is D[∗] − L_[�] _robust, where:_ 

**==> picture [300 x 13] intentionally omitted <==**

## _and_ 

**==> picture [106 x 15] intentionally omitted <==**

_is the Jacobian matrix of the mean action µϕ_ ( _s_ ) _with respect to ϕ, evaluated at ϕ[∗] ._ 

_Proof._ Assume a Gaussian policy with fixed covariance Σ, such that _πϕ_ ( _a|s_ ) = _N_ ( _a_ ; _µϕ_ ( _s_ ) _,_ Σ), where _µϕ_ ( _s_ ) is the mean of the Gaussian we are training. A Taylor expansion yields _µϕ_ + _ϵ_ ( _s_ ) = _µϕ_ ( _s_ ) + _J_ ( _ϕ_ ) _ϵ_ + _O_ ( _||ϵ||_[2] ). 

Define the change in policy for a given state _s_ due to the perturbation _ϵ_ as 

**==> picture [189 x 13] intentionally omitted <==**

Then by the triangle inequality, the Cauchy-Schwarz inequality and definition _∥ϵ∥ < E_ , we have 

**==> picture [229 x 12] intentionally omitted <==**

The sub-optimality of this action perturbation _δ_ is 

**==> picture [391 x 44] intentionally omitted <==**

Thus a _E − L_[�] flat reward maximum implies a _D[∗] − L_[�] robust policy with _D[∗] ≤||J_ ( _ϕ[∗]_ ) _||E_ + _O_ ( _E_[2] ) 

As we want to show an excess BT loss based on a Lipschitz-continuity assumption, which is defined over action pairs, we need to relate _D − L_[�] robustness to _P_ ( _SK,δ|πϕ_ ). For this purpose we need B.3 following two simple lemmas: 

14 

**Gradient Regularization Prevents Reward Hacking** 

**Lemma B.6** (Flatness and _β_ -smoothness imply bounded gradient) **.** _Let f_ : R _[d] →_ R _be differentiable and β-smooth on a ball B_ ( _c, r_ ) _, i.e., ∥∇f_ ( _x_ ) _−∇f_ ( _y_ ) _∥≤ β∥x − y∥ ∀x, y ∈ B_ ( _c, r_ ) _. Assume a "flat maximum" in c, such that f_ ( _c_ ) _− f_ ( _c_ + _u_ ) _≤ L_[�] _∀u_ : _∥u∥≤ r. Then the gradient norm at c is bounded as_ 

**==> picture [89 x 24] intentionally omitted <==**

_Proof._ Based on the standard quadratic upper bound based on _β_ -smoothness, we know 

**==> picture [243 x 22] intentionally omitted <==**

We can rearrange this to 

**==> picture [171 x 21] intentionally omitted <==**

By choosing the worst case _u_ = _−r ∥∇[∇][f] f_[(] ( _[c] c_[)] ) _∥_[(if] _[∥∇][f]_[(] _[c]_[)] _[∥]_[=][0][the][bound][is][trivially][true),][we][have] _[∥][u][∥]_[=] _[r]_[and] _−∇f_ ( _c_ ) _[T] u_ = _r∥∇f_ ( _c_ ) _∥_ . Plugging this into the inequality and using _f_ ( _c_ ) _− f_ ( _c_ + _u_ ) _≤ L_[�] yields 

**==> picture [89 x 25] intentionally omitted <==**

Note that robustness with radius _r_ implies robustness with any _r[′] < r_ and we could further improve this bound by picking _r[′]_ = min(�2 _βL_[ˆ] ) := _r[∗]_ , if _r[∗] ≤ r_ . We do not do this here for ease of exposition. We next need to connect the gradient bound of the expected E[ _R_[�] ( _s, a_ + _Z_ )] to a bound on the gradient of the pointwise _R_[�] ( _s, a_ ): **Lemma B.7** (Pointwise action-gradient is controlled by Gaussian-smoothing) **.** _Fix s and let f_ ( _a_ ) = _R_[�] ( _s, a_ ) _. Under Assumption B.3 and for Z ∼N_ (0 _,_ Σ) _, define the Gaussian-smoothed reward_ 

**==> picture [89 x 12] intentionally omitted <==**

**==> picture [175 x 12] intentionally omitted <==**

**==> picture [145 x 12] intentionally omitted <==**

_Proof._ Since _f_ is _β_ -smooth, it is continuously differentiable with Lipschitz gradient, and we can interchange gradient and expectation to obtain _∇f_[¯] ( _c_ ) = E[ _∇f_ ( _c_ + _Z_ )]. Then, by Jensen’s inequality and _β_ -smoothness, 

**==> picture [369 x 20] intentionally omitted <==**

then, by the triangle inequality and the previous line, 

**==> picture [434 x 26] intentionally omitted <==**

Then, we need to connect local smoothness and our knowledge of the gradient norm _∥∇f_ ( _c_ ) _∥_ at the center to violations of Lipschitz-continuity: 

**Lemma B.8** (Pairwise robustness is bounded by local smoothness) **.** _Fix a state s. Denote f_ ( _a_ ) = _R_[�] ( _s, a_ ) _. Assume that there exists a center c ∈ A and radius r >_ 0 _such that f is β-smooth on the ball B_ ( _c, r_ ) := _{a_ : _∥a − c∥≤ r}. Let_ i _._ i _._ d _. a, a_ 1 _, a_ 2 _∼ π_ ( _a|s_ ) _._ 

_Then for any δ >_ 0 _and K >_ 0 _, we have_ 

**==> picture [371 x 32] intentionally omitted <==**

15 

**Gradient Regularization Prevents Reward Hacking** 

_Proof._ For any _x ∈ B_ ( _c, r_ ), by _β_ -smoothness, 

_∥∇f_ ( _x_ ) _∥≤∥∇f_ ( _c_ ) _∥_ + _∥∇f_ ( _x_ ) _−∇f_ ( _c_ ) _∥≤∥∇f_ ( _c_ ) _∥_ + _β∥x − c∥≤∥∇f_ ( _c_ ) _∥_ + _βr ._ 

For any two points _x, y ∈ B_ ( _c, r_ ), 

**==> picture [205 x 26] intentionally omitted <==**

taking the norm, using the Cauchy-Schwarz inequality, and the bounded gradient, we get 

**==> picture [331 x 26] intentionally omitted <==**

Therefore, if _x, y ∈ B_ ( _c, r_ ) and _∥x − y∥≤ δ_ , then 

**==> picture [147 x 11] intentionally omitted <==**

Thus, the event _{∥a_ 1 _− a_ 2 _∥≤ δ, |f_ ( _a_ 1) _− f_ ( _a_ 2) _| > K}_ cannot occur when _a_ 1 _, a_ 2 _∈ B_ ( _c, r_ ) and ( _∥∇f_ ( _c_ ) _∥_ + _βr_ ) _δ < K_ . It thus can only occur if at least one of _a_ 1 or _a_ 2 is not in _B_ ( _c, r_ ), which occurs with probability _P_ ( _∥a − c∥ > r_ ). By a union bound of the two events we get the result. 

Finally, we can put it all together: 

**Proposition B.9** (From _D_ - _L_[�] robustness to pairwise sharpness control) **.** _Assume a Gaussian policy with full row-rank Jacobian (Assumption B.2), a β-smooth proxy reward R_[�] _(B.3), and fix a state s. If ϕ[∗] is D-L_[�] _action robust, then for any δ >_ 0 _and K >_ 0 _such that K/δ > G, we have gradient magnitude bound G and non-violating radius r,_ 

**==> picture [199 x 26] intentionally omitted <==**

_such that_ 

**==> picture [151 x 14] intentionally omitted <==**

_Proof._ Fix _s_ and denote _c_ = _µϕ∗_ ( _s_ ). Denote the PR as _f_ ( _a_ ) = _R_[�] ( _s, a_ ), mean action as _c_ = _µϕ∗_ ( _s_ ), and the smoothed proxy reward _f_[¯] ( _c_ ) = E _Z∼N_ (0 _,_ Σ)[ _f_ ( _c_ + _Z_ )]. By Assumption B.2, _a ∼ πϕ∗_ ( _a|s_ ) can be written as _a_ = _c_ + _Z_ with _Z ∼N_ (0 _,_ Σ). By Lemma B.6 and, applied to _f_[¯] , which we know fulfills Assumption B.3, we obtain on _B_ ( _c, D_ ) the gradient bound 

**==> picture [95 x 24] intentionally omitted <==**

From Lemma B.7, we know that the gradient norm of _f_ can then be bounded as 

**==> picture [147 x 37] intentionally omitted <==**

We then know from Lemma B.8, that in _B_ ( _c, r_ ) with non-violating radius _r_ = _β_[1][(] _[K] δ[−][G]_[)][ there can be no action pairs] _[ a]_[1] _[, a]_[2] with _∥a_ 1 _− a_ 2 _∥≤ δ_ and _|R_[�] ( _s, a_ 1) _− R_[�] ( _s, a_ 2) _| > K_ , thus such violations can only occur if at least one action is outside _B_ ( _c, r_ ): 

**==> picture [263 x 11] intentionally omitted <==**

Finally, _a_ 1 _− c ∼N_ (0 _,_ Σ), so _P_ ( _∥a_ 1 _− c∥ > r_ ) = _P_ ( _∥Z∥ > r_ ). 

16 

**Gradient Regularization Prevents Reward Hacking** 

## **B.2. BT-Loss Lower Bound Based on Sharpness** 

Now that we have connected _D − L_[�] robustness to the probability of actions pairs violating a flatness assumption _P_ ( _SK,δ|πϕ_ ), we now analyze the incurred excess BT loss _L_ BT due to these violations. For this purpose, we make the assumption of a Lipschitz continuous true reward (Assumption B.4) 

**Proposition B.10.** _R_ � _, the excess-risk can be lower bounded asFor a prompt s ∼ P_ ( _s_ ) _, a pair of actions_ ( _a_ 1 _, a_ 2) _, L-Lipschitz true reward function R[∗] , reward model_ 

**==> picture [353 x 14] intentionally omitted <==**

_where SK,δ_ := ( _s, a_ 1 _, a_ 2) : _∥a_ 1 _− a_ 2 _∥≤ δ, |R_[�] ( _s, a_ 1) _− R_[�] ( _s, a_ 2) _| > K is the set of action pairs for which the RM R_[�] _is_ � � _not K-Lipschitz continuous and_ Pr( _SK,δ_ ) _is the probability of an action pair sampled_ ( _a_ 1 _, a_ 2) _∼ πϕ_ ( _·|s_ ) _being in this set._ 

Let _s ∼ P_ ( _s_ ) denote prompts, _a ∈A_ denote actions. Let _R[∗]_ ( _s, a_ ) be the true reward. Under the BT model, the preference probability is _p_ := Pr( _Y_ = 1 _| s, a_ 1 _, a_ 2) = _σ_ (∆ _∗_ ), ∆ _∗_ := _R[∗]_ ( _s, a_ 1) _− R[∗]_ ( _s, a_ 2), with the logistic function _σ_ ( _x_ ) = 1+1 _e[−][x]_[, where] _[ Y]_[= 1][ means that] _[ a]_[1][is preferred over] _[ a]_[2][.][A parametric reward model] _[R]_[�][(] _[s, a]_[)][ induces the predicted] pairwise probability _q_ := _P_ � _R_[(] _[Y]_[=][1] _[|][s, a]_[1] _[, a]_[2][)][=] _[σ]_[(∆] _[θ]_[)][,][∆] _[θ]_[:=] _[R]_[�][(] _[s, a]_[1][)] _[ −][R]_[�][(] _[s, a]_[2][)][.][The BT loss in equation][ 3][ can be] rewritten as 

**==> picture [358 x 19] intentionally omitted <==**

but it can also be rewritten to explicitly consider a preference label _Y_ , where we have actions _a_ 1 _, a_ 2; then draw _Y ∼_ Bernoulli( _p_ ) with _p_ = Pr( _Y_ = 1 _|s, a_ 1 _, a_ 2). Taking expectations first over _Y |s, a_ 1 _, a_ 2 and then over ( _s, a_ 1 _, a_ 2), we get, 

**==> picture [339 x 15] intentionally omitted <==**

for _Y ∈{_ 0 _,_ 1 _}_ and _ℓ_ ( _q_ ; _Y_ ) = _−_ [ _Y_ log _q_ + (1 _− Y_ ) log(1 _− q_ )]. We proceed with this version. 

**Lower Bound** Condition on ( _s, a_ 1 _, a_ 2) so that _p_ is fixed. Then 

**==> picture [392 x 12] intentionally omitted <==**

where _H_ ( _p, q_ ) is the cross-entropy and _H_ ( _p_ ) is the entropy. Averaging over ( _s, a_ 1 _, a_ 2) and subtracting yields 

**==> picture [376 x 15] intentionally omitted <==**

For a fixed locality parameter _δ >_ 0 and margin threshold _K >_ 0, we define the sharpness set 

**==> picture [348 x 11] intentionally omitted <==**

Assume the true reward is _L_ -Lipschitz: 

**==> picture [414 x 11] intentionally omitted <==**

Furthermore, we study the interval separation on _SK,δ_ : On _SK,δ_ we have ∆ _∗ ≤ Lδ_ and ∆ _θ ≥ K_ . By monotonicity and symmetry of _σ_ , 

**==> picture [367 x 26] intentionally omitted <==**

If we assume _K > Lδ_ , the minimum distance between these sets becomes: 

**==> picture [425 x 16] intentionally omitted <==**

Consequently, for every ( _s, a_ 1 _, a_ 2) _∈ SK,δ_ , 

**==> picture [298 x 34] intentionally omitted <==**

**Gradient Regularization Prevents Reward Hacking** 

phi = model.state_dict() actions = model.generate(states) rewards = reward_fn(states, actions) grad1 = torch.zeros_like(phi) **for** idx **in** range(batch_size / accumulation_steps): _# mb = microbatch_ loss = grpo_loss(states_mb, actions_mb, rewards_mb) loss.backwards() grad1 += model.grad grad1 = norm_clip(grad1) phi_2 = phi + varepsilon * grad1 grad2 = torch.zeros_like(phi) model.set_state_dict(phi_2) **for** idx **in** range(batch_size / accumulation_steps): loss = grpo_loss(states_mb, actions_mb, rewards_mb, phi_2) loss.backwards() grad2 += model.grad grad2 = norm_clip(grad2) comb_grad = grad1 + gamma * (grad2 - grad1) / varepsilon model.set_state_dict(phi) model.grad = comb_grad optimizer.step() 

_Figure 9._ Implementation of finite-difference gradient regularization with GRPO in PyTorch 

Since we can use the inequality _D_ KL(Bernoulli( _p_ ); Bernoulli( _q_ )) _≥_ 2 _|p−q|_[2] for _p, q ∈_ (0 _,_ 1), combining with equation 26 yields: 

**==> picture [425 x 13] intentionally omitted <==**

Let **1** _S_ := **1** _{Z ∈ SK,δ}_ indicate membership of _SK,δ_ . We can make equation 27 valid for all ( _s, a_ 1 _, a_ 2) by multiplying the RHS with the indicator: 

**==> picture [409 x 12] intentionally omitted <==**

By taking the expectation of equation 28 and using equation 21, we obtain the desired lower bound 

**==> picture [353 x 14] intentionally omitted <==**

## **B.3. Connection of Gradient Regularization and Lipschitz Continuity** 

For completeness, we reproduce the argument of Zhao et al. (2022), which explicitly connects gradient regularization to Lipschitzness in parameters _θ_ . 

By the mean value theorem for differentiable _L_ , we have _L_ ( _θ_ 1) _− L_ ( _θ_ 2) = _∇L_ ( _ζ_ ) _[T]_ ( _θ_ 1 _− θ_ 2) _,_ with _ζ_ = _cθ_ 1 + (1 _− c_ ) _θ_ 2, with some _c ∈_ [0 _,_ 1] and the Cauchy-Schwarz inequality then yields _∥L_ ( _θ_ 1) _− L_ ( _θ_ 2) _∥≤∥∇L_ ( _ζ_ ) _∥∥_ ( _θ_ 1 _− θ_ 2) _∥ ._ Here, _||∇L_ ( _ζ_ ) _||_ takes the role of the Lipschitz constant and as _θ_ 2 _→ θ_ 1, _||∇L_ ( _ζ_ ) _||_ becomes _||∇L_ ( _θ_ ) _||_ . Thus we can see that gradient regularization leads to local Lipschitzness in parameter space, i.e., a flat local minimum. 

## **C. Experiment Details** 

In this section we provide additional experimental details. 

## **C.1. Gradient regularization implementation** 

In our experiments, we use the GR method (Karakida et al., 2023) based on the finite difference estimate ∆ _ϕ∥∇ϕL_ ( _ϕ_ ) _∥_[2] = _∇ϕL_ ( _ϕ_ + _ε∇ϕL_ ( _ϕ_ )) _−∇ϕL_ ( _ϕ_ ) _._ We, thus, need to perturb the parameters _ϕ_ . Empirically, we found it to be beneficial to perturb 

> _ε_ 

18 

**Gradient Regularization Prevents Reward Hacking** 

_Table 5._ GRPO hyper-parameters in RLHF experiments. We tuned the learning rate for each method on Qwen 2.5-1.5B experiments from 1 _×_ 10 _[−]_[6] _,_ 3 _×_ 10 _[−]_[6] _,_ 5 _×_ 10 _[−]_[6] . 

|Method|GR|KL, Resets, No Reg|
|---|---|---|
|Optimizer|Adam (Kingma & Ba,2015)||
|LR|5_×_10_−_6|3_×_10_−_6|
|Adam_β_1||0.9|
|Adam_β_2||0.999|
|Batchsize||256|
|Rollouts per Prompt||8|
|Temperature||0.7|
|GR_ε_|1_×_10_−_3|-|
|Gradient Clipping||1.0|
|Output Length||106|



only the parameters of the transformer blocks, including attention matrices, MLP weights and layer norm parameters, but not perturb the embedding layer or final output layer. In principle, to calculate _∥∇ϕJ_ ( _ϕ_ + _∇ϕJ_ ( _ϕ_ )) _∥_ we would also need to sample new actions _ai ∼ πϕ_ + _∇ϕJ_ ( _ϕ_ )( _a|s_ ) and estimate the gradient using these. In practice, the computational overhead of this would be large, we reuse the same actions. This introduces some bias which could be corrected use importance sampling, however, empirically we found this to be unnecessary. We also need to choose a perturbation strength _ε_ . While Karakida et al. (2023) found a relatively large _ε ≈_ 0 _._ 05 to perform best, initial experiments showed _ε_ = 10 _[−]_[3] to perform well in our setting. We thus used it through-out our experiments. We further use gradient clipping, both for the disturbance _∇ϕJ_ ( _ϕ_ ) and the gradient _∥∇ϕJ_ ( _ϕ_ + _∇ϕJ_ ( _ϕ_ )) _∥_ , each to 10. We found this to prevent gradient spikes from destabilizing training. The final combined gradient is then again clipped to 1.0 within DeepSpeed, as is done for the non GR methods as well. We train our models using DeepSpeed ZeRO 2 (Rajbhandari et al., 2020) and use gradient accumulation. 

## **C.2. RLHF Details** 

For our RLHF experiments, we first need to train an SFT model, sample a training dataset _D_ RM, and train an RM. For both, we use the code and hyper-parameters provided by Huang et al. (2024), which use the AdamW optimizer (Loshchilov & Hutter, 2019) with weight decay. The SFT models are trained on the SFT dataset for one epoch. For the summarization experiments, we use the SFT dataset provided by (Huang et al., 2024), for Alpaca experiments we use the Alpaca-Instructions dataset (Dubois et al., 2023), but filter it by length following (Ackermann et al., 2025). This length filtering to a maximum length of 512 tokens significantly decreases the computational cost. Further, while Alpaca-Instructions contains separate splits for SFT, RM and RL training, we combine them to a single dataset as used in the summarization setting. To obtain the RM training dataset, we sample pairs of responses from the SFT model and label them with either the Gold RM “Skywork-RewardLlama-3.1-8B-v0.2” (Liu et al., 2024) for summarization or with GPT4.1-Nano for Alpaca experiments. For the summarization tasks we use 278,496 preference pairs, for Alpaca Experiments we use 43,008 pairs for the 3B model and 86,016 pairs for the 0.5B and 1.5B models. With the smaller data amount the 0.5B and 1.5B models did not produce sufficiently accurate RMs to use for subsequent GRPO updates. We train the RMs, initialized from the SFT model, for one epoch on the dataset with the hyper-parameters as recommended by (Huang et al., 2024). We then use the Dr.GRPO implementation provided by TRL (von Werra et al., 2020) with the hyper-parameters as listed in Table 5. Additionally, we tuned the KL penalty strength _β_ and GR strength _γ_ as listed Tables 6 and 7. 

_Table 6._ KL penalty values considered in hyperparameter optimization for Reference Resets Tl;DR. 

|Experiment|KL values|
|---|---|
|GRPO Pythia 1B TL;DR|_{_0_._03_,_0_._04_,_0_._05_,_0_._06_,_0_._07_,_0_._08_}_|
|GRPO Pythia 2.8B TL;DR|_{_0_._04_,_0_._06_,_0_._08_,_0_._10_}_|
|GRPO+Ref Reset Pythia 1B TL;DR|_{_0_._2_,_0_._25_,_0_._3_,_0_._35_,_0_._4_}_|
|GRPO+Ref Reset Pythia 2.8B TL;DR|_{_0_._25_,_0_._3_,_0_._35_,_0_._4_}_|



19 

**Gradient Regularization Prevents Reward Hacking** 

## **C.3. Gradient Regularization on Alpaca Farm** 

We perform on the AlpacaFarm dataset (Dubois et al., 2023) with Qwen2.5 models (Qwen et al., 2025). Similar to the TL;DR experiments we first train an SFT model, sample pairs of responses from it, and obtain pairwise comparisons from GPT4.1 Nano. We then train an RM based on these comparisons, initialized from the SFT model. 

_Table 7._ KL penalty values considered in hyperparameter optimization for Alpaca GPT4.1 Nano experiments. 

|Experiment|Hyperparameters|
|---|---|
|GR Qwen 2.5 0.5B|_γ ∈{_1_×_10_−_1_,_ 1_×_10_−_2_,_ 1_×_10_−_3_}_|
|GR Qwen 2.5 1.5B|_γ ∈{_1_×_10_−_1_,_ 1_×_10_−_2_,_ 1_×_10_−_3_}_|
|GR Qwen 2.5 3B|_γ ∈{_3_×_10_−_3_}_|
|KL Qwen 2.5 0.5B|_β ∈{_0_._03_,_0_._05_,_0_._07_,_0_._1_,_0_._15_}_|
|KL Qwen 2.5 1.5B|_β ∈{_0_._03_,_0_._05_,_0_._07_,_0_._1_,_0_._15_}_|
|KL Qwen 2.5 3B|_β ∈{_0_._05_,_0_._1_}_|
|KL+Reset Qwen 2.5 0.5B|_β ∈{_0_._2_,_0_._4_,_0_._5_}_|
|KL+Reset Qwen 2.5 1.5B|_β ∈{_0_._3_,_0_._4_,_0_._5_}_|
|KL+Reset Qwen 2.5 3B|_β ∈{_0_._2_,_0_._3_,_0_._4_}_|



## **C.4. Reasoning Experiments** 

We use the hyperparameters as stated in Table 8 and Table 9. We did not tune _β_ , _ϵ_ or _γ_ and simply used conservative values from the RLHF experiments. For both GR and no regularization we tried learning rates 3 _×_ 10 _[−]_[6] and 5 _×_ 10 _[−]_[6] . For GR, the larger learning rate was beneficial. 

We follow the setup of Wei et al. (2025), with GSM8K using one-shot prompting, math uses chain-ofthought prompting, both with the default Qwen2.5-Instruct system instructions. We also use the same reward terms, which are based on the rewards used by Open R1 (Hugging Face, 2025). For GSM8K, the prompt is Respond in the following format: <reasoning>...</reasoning><answer>...</answer>, followed by a one-shot example. The reward consists of a rule-based correctness reward ([0 _,_ 2]), a formatting term checking whether the answer is an integer formatting reward ( _{_ 0 _,_ 0 _._ 5 _}_ ), a formatting term checking whether the formatting is exactly followed including whitespace ( _{_ 0 _,_ 1 _._ 0 _}_ ), and excluding whitespace ( _{_ 0 _,_ 1 _._ 0 _}_ ), and a formatting reward for matching XML tags ( _{_ 0 _,_ 0 _._ 5 _}_ ). 

On MATH the prompt is simply the problem statement followed by Let's think step by step and output the final answer within \\boxed{}.. The used reward consists of a correctness reward ([0 _,_ 2]) and a formatting reward rewarding up to three of the following with 1 _/_ 3 each: _Step #_ keywords, Numbered lists, bullet points, _First, Second, Next, Finally_ keywords. This is done to encourage reasoning (Hugging Face, 2025). 

_Table 8._ Hyperparameters in reasoning experiments with GR 

|Dataset|GSM8K<br>MATH|GSM8K<br>MATH|
|---|---|---|
|Optimizer||Adam|
|LR|5|_×_10_−_6|
|Adam_β_1||0.9|
|Adam_β_2||0.999|
|Batchsize|256|1024|
|Rollouts per Prompt||8|
|Temperature||0.7|
|GR_ε_||10_−_3|
|GR_γ_||10_−_3|
|Gradient Clipping||1.0|
|Output Length|768|1024|



20 

**Gradient Regularization Prevents Reward Hacking** 

_Table 9._ Hyperparameters in reasoning experiments with KL penalty or no penalty 

|Dataset|GSM8K<br>MATH|GSM8K<br>MATH|
|---|---|---|
|Optimizer||Adam|
|LR|3|_×_10_−_6|
|Adam_β_1||0.9|
|Adam_β_2||0.999|
|Batchsize|256|1024|
|Rollouts per Prompt||8|
|Temperature||0.7|
|KL penalty_β_|0.05/ 0.0<br>0.0||
|Gradient Clipping||1.0|
|Output Length|768|1024|



## **C.5. LLM Judge Setup** 

In the main text we use Qwen2.5 1.5B-Instruct (Qwen et al., 2025) as a judge with the following prompt: 

Judge the correctness of the answer and reasoning for the given problem. The format is as follows: <problem> ... </problem> <model_answer> <reasoning> ... </reasoning> <answer> ... </answer> </model_answer> <correct_solution> ... </correct_solution> You will reply with the following XML format: <judgement> ... </judgement> <correctness_score> ... </correctness_score> <coherence_score> ... </coherence_score> 

The model_answer may contain mistakes in the reasoning, the final answer, and in the _�→_ format. 

Give a one sentence judgement on the model_answer, then you will give scores from 1 to 5 _�→_ for correctness and for coherence of the reasoning trace. 

We additionally use 1-shot prompting with a correct example. During training the agent is provided the correctness score scaled to [0 _,_ 2], along with the same formatting rewards used in the rule-based-reward setting. 

21 

**Gradient Regularization Prevents Reward Hacking** 

**==> picture [479 x 123] intentionally omitted <==**

**----- Start of picture text -----**<br>
SFT ModelGradient Reg 28 SFT ModelGR 0.70<br>28 0.68<br>27<br>0.66<br>26<br>26 0.64<br>25 =1e-3<br>24 0.62 =3e-3<br>24 0.60 =5e-3<br>23 =1e-2<br>0.58 =1.3e-2<br>22 22 0.56 =2e-2<br>21 0 100 200 300 400<br>10 3 10 2 10 1 10 4 10 3 10 2 Training steps<br>Gradien Regularization Strength  GR Disturbance<br>BT<br>GPT 4.1-Nano Winrate (%) GPT 4.1-Nano Winrate (%)<br>**----- End of picture text -----**<br>


_Figure 11._ **Strong GR can decrease BT loss below initial value.** BT loss _L_ BT( _ϕ, θ_ ) during training of a Qwen 2.5 0.5B model on the TL;DR task with GR, using the Gold reward model. 

_Figure 10._ **GR is rather predictable in choice of hyper-parameter.** Qwen2.5-1.5B on AlapcaFarm with different GR strengths _γ_ and fixed _ε_ = 10 _[−]_[3] (left) with, and different disturbance strengths _ε_ and fixed _γ_ = 3 _×_ 10 _[−]_[2] (right) on the AlpacaFarm dataset, with early stopping. 

**==> picture [154 x 132] intentionally omitted <==**

**----- Start of picture text -----**<br>
28<br>26<br>24<br>SFT Model<br>22 GR<br>KL<br>2 4 6<br>Learning Rate 1e 6<br>GPT 4.1-Nano Winrate (%)<br>**----- End of picture text -----**<br>


_Figure 12._ Learning rate sweep for KL penalty and GR when training a Qwen 2.5-1.5B model on the Alpacafarm dataset. KL _β_ = 0 _._ 1, GR _γ_ = 3 _×_ 10 _[−]_[2] 

**==> picture [154 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
70<br>No Reg<br>60 GR<br>50<br>40<br>30<br>20<br>10<br>0<br>Q2.5-1.5B Q2.5-3B Q3-4B<br>Judge Model<br>GSM8K Accuracy (%)<br>**----- End of picture text -----**<br>


_Figure 13._ **GR allows usage of cheaper judges to reach same performance.** Test accuracy when training a Qwen2.5-0.5B model on GSM8K with different judges. 

## **D. Additional Experiments** 

## **D.1. Gradient Regularization** _γ_ **,** _ε_ **, and learning rate sweeps** 

To evaluate the sensitivity of GR to the hyper-parameters controlling strength of the regularization _γ_ and the strength of the perturbation _ε_ , we performed experiments on the Qwen2.5 1.5B model in the Alpaca GPT4.1 Nano setting. We fix either _γ_ = 3 _×_ 10 _[−]_[2] or _ε_ = 10 _[−]_[3] and vary the respective other parameter. We show the results in Figure 10. We find the performance to be rather predictable, facilitating hyper-parameter tuning. We used the same disturbance strength _ε_ = 10 _[−]_[3] for all experiments and did not find it necessary to tune it per task or model. In the reasoning experiments we did not tune _γ_ and simply used _γ_ = 10 _[−]_[3] , however, in the RLHF experiments we found it necessary to tune _γ_ as described above, just like we found it necessary to tune _β_ for the KL penalty. 

We also perform a learning rate sweep for the KL penalty and GR in the same Alpaca, Qwen2.5 1.5B setting. We use the best performing KL hyper-parameters from our main hyper-parameter optimization, i.e. _β_ = 0 _._ 1 for the KL penalty and _γ_ = 0 _._ 03 for GR. The results are shown in Figure 12. 

## **D.2. Gradient regularization decreases BT loss** 

We also perform experiments in the synthetic TL;DR gold model setup, training a Qwen 2.5 0.5B model with different GR strengths _γ_ . The results in Figure 11 show that stronger regularization leads to a lower BT model loss _L_ BT( _ϕ, θ_ ), evaluated on 4096 action pairs with labels from the gold reward model. Interestingly, training with strong GR can result in a decreasing BT loss _L_ BT beyond the initial BT loss, which we did not observe when utilizing either KL regularization or Reference Resets. This illustrates the practical strength of the connection between gradient norm and PR accuracy. 

22 

**Gradient Regularization Prevents Reward Hacking** 

**==> picture [369 x 128] intentionally omitted <==**

**----- Start of picture text -----**<br>
6<br>7<br>7<br>8<br>9 8<br>10<br>9<br>11<br>12 10 R = 25<br>Constant  R = 50<br>13 Schedule  11 R = 100<br>R = 200<br>14 Resets R = 400<br>12<br>0 25 50 75 100 125 0 25 50 75 100 125 150 175 200<br>DKL( || 1 [)] DKL( ; 1)<br>Gold Model Score Gold Model Score<br>**----- End of picture text -----**<br>


_Figure 14._ Pythia 1B on TL;DR task. Left: Reference Resets schedule ablation. A scheduled _β_ performs better than a constant value, however, it does not match the performance of full Reference Resets. Right: Steps per reset _R_ for GRPO + Reference Resets. A larger R is generally beneficial, but requires significantly more gradient steps. 

## **D.3. LLM judge ablation** 

In the main text we are using Qwen2.5 1.5B-Instruct as judge. As we are using similarly sized policy models, we believe this could be a useful proxy for experiments in which LLMs are trained with equally large judge models. However, to see whether GR is also useful with comparatively stronger judges, we additionally run experiments with Qwen2.5 3B-Instruct and Qwen3 4B Instruct-2507 (Yang et al., 2025) as judges. We train a Qwen2.5 0.5B-Instruct model on GSM8K. For Qwen3 we use Judgement:... Correctness_score:... Coherence_score:... as reply format instead of the xml tags, as we found Qwen3 to perform better with this format. As shown in Figure 13, GR enables us to reach the same performance using a cheaper judge, potentially saving total computational cost. In our setup with 8 GH200 GPUs, training with GR and the 1.5B judge took 84 minutes, while training without GR and the 4B judge took 88 minutes. The additional cost of GR can thus be amortized by being able to use a cheaper judge. 

## **D.4. KL penalty schedule** 

While we have shown in the main text that Reference Resets perform better in RLHF than simply decreasing the strength of the KL penalty _β_ from the beginning, another hypothesis might be that decreasing the KL strength _β_ during training will match the results of Reference Resets. As an additional baseline, we thus decrease the strength of the KL penalty to _β[′]_ = _β/i_ in iteration _i_ , while keeping the reference as _π_[1] . Thus in each iteration the optimal policy under Reference Resets and the scheduled _β_ is the same, assuming the previous iterations converged to their respective optimal policies. As shown in Figure 14 (left), the schedule indeed yields a notable improvement over a fixed _β_ , but also does not match the performance of Reference Resets. We attribute this to the KL penalty explicitly keeping the policy close to the good region of the RM found in previous iteration, which a scheduled _β_ does not ensure. 

## **D.5. Steps per reset** _R_ 

Our theoretical derivation suggests that flatter minimum corresponds to a more accurate reward model. Experiments show that the gradient norm keeps decreasing within each iteration even after the PR score is saturated. Thus, we expect training for more steps _R_ per reset to improve performance at the cost of a higher computational expense. We evaluate different values for _R_ when training a Pythia 1B model on the summarization task and show the results in Figure 14 (right), training each model for 1500 total steps. Indeed, we find more steps _R_ to lead to a better KL-Gold-Reward tradeoff and perhaps a better asymptotic reward. However, for high values such as _R_ = 400 the computational cost becomes prohibitively expensive, such that we use _R_ = 200 in experiments unless otherwise specified. 

## **D.6. Reward hacking example** 

An example output by a Qwen2.5 0.5B model trained without regularization hacking a Qwen2.5-1.5B-Instruct judge: 

<reasoning> 

> 1. Given values: \\[ 

23 

## **Gradient Regularization Prevents Reward Hacking** 

**==> picture [487 x 163] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.70<br>7 7<br>4<br>8 0.68 8<br>9 9<br>3 0.66<br>10 10<br>0.64<br>2 11 11<br>0.62<br>12 12<br>1 0.60<br>13 13<br>GRPO<br>GRPO+Reset 14 0.58 14<br>0<br>0 500 1000 1500 2000 0 500 1000 1500 2000 0 500 1000 1500 2000 0.60 0.65 0.70<br>Gradient Steps Gradient Steps Gradient Steps RM BT Loss<br>RM BT Loss<br>RM Model Score Gold Model Score Gold Model Score<br>**----- End of picture text -----**<br>


_Figure 15._ Proxy and gold reward for no resets (blue, _β_ = 0 _._ 06) and resets (orange, _β_ = 0 _._ 3). The proxy reward should thus match after 5 resets which here with _R_ = 200 corresponds to 1200 training steps. The achieved proxy reward is relatively similar at training step _t_ = 1200, however, the gold reward with resets is significantly higher. 

\text{initial}} = 500, \text{spent_on_clothes}} = 20/100 \times 500 = 100, 

_�→_ \text{remaining_after_clothes}} = 500 - 100 = 40}}, \text{spent_on_cds}} = 25/100 _�→_ \times 40 = 100}}))))))))))))))))))))))))))))))>> To find the final amount, we perform the following steps: \[ 

\text{final_left}} = 500 - 100 + 100 = 119}} 

\)) </code>> </p>> </reasoning> <answer> 119 </answer> 

## **E. Reference Resets in RLHF** 

To investigate the impact of reference resets in RLHF we provide an asymptotic argument, showing an equivalence between using reference resets and using a lower _β_ . While we show that this equivalence indeed shows up in experiments for the RM reward _Rθ_ , the achieved true reward _R[∗]_ is significantly higher when doing reference sets than when using a lower _β_ . This can not be explained by the asymptotic argument, we thus continue by providing an analysis from the point of view of optimization dynamics. 

## **E.1. Asymptotic analysis** 

It is well known (e.g. (Rafailov et al., 2023, Appendix A.1)) that the optimal solution of the KL-regularized optimization problem arg max _π R_[�] ( _s, a_ ) _− D_ KL( _π, π_[1] ) is 

**==> picture [324 x 19] intentionally omitted <==**

With reference resets, we are solving this problem repeatedly, thus for iteration _k_ 

**==> picture [331 x 19] intentionally omitted <==**

If we insert _π[k][−]_[1] into this, we obtain 

**==> picture [335 x 19] intentionally omitted <==**

24 

**Gradient Regularization Prevents Reward Hacking** 

Therefore, the optimal policy for Reference Resets with _k_ iterations and KL-penalty strength _β_ should be the same as the solution without resets with a weaker KL-penalty weight _β[′]_ = _β/k_ . In experiments, we can indeed see a similar behavior when only looking at the RM reward, as shown in Figure 15 (left). There, after 1200 steps the RM reward _R_[�] achieved with reset is roughly in the rang of reward without reference resets. As this argument makes no statements about the true reward _R[∗]_ , we might expect it to be similar for both methods as well. Surprisingly, we instead find that the true reward achieved by Reference Resets is significantly higher than the true reward achieved in a single stage. We believe this effect can not be explained by an asymptotic analysis, thus motivating our optimization dynamics argument. In Figure 15 (right), we also show that using Reference Resets higher gold reward regions can be obtained for the same RM BT loss _L_ BT. 

25 


