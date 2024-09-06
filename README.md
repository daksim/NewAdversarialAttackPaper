# Latest Adversarial Attack Papers
**update at 2024-09-06 15:32:00**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

cs.CR

This paper completes its earlier vision paper, available at  arXiv:2402.15727. Updated to the latest analysis and results

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2406.05498v2) [paper-pdf](http://arxiv.org/pdf/2406.05498v2)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform six state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to adaptive jailbreaks and prompt injections.



## **2. How to Train your Antivirus: RL-based Hardening through the Problem-Space**

cs.CR

20 pages,4 figures

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2402.19027v2) [paper-pdf](http://arxiv.org/pdf/2402.19027v2)

**Authors**: Ilias Tsingenopoulos, Jacopo Cortellazzi, Branislav Bošanský, Simone Aonzo, Davy Preuveneers, Wouter Joosen, Fabio Pierazzi, Lorenzo Cavallaro

**Abstract**: ML-based malware detection on dynamic analysis reports is vulnerable to both evasion and spurious correlations. In this work, we investigate a specific ML architecture employed in the pipeline of a widely-known commercial antivirus company, with the goal to harden it against adversarial malware. Adversarial training, the sole defensive technique that can confer empirical robustness, is not applicable out of the box in this domain, for the principal reason that gradient-based perturbations rarely map back to feasible problem-space programs. We introduce a novel Reinforcement Learning approach for constructing adversarial examples, a constituent part of adversarially training a model against evasion. Our approach comes with multiple advantages. It performs modifications that are feasible in the problem-space, and only those; thus it circumvents the inverse mapping problem. It also makes possible to provide theoretical guarantees on the robustness of the model against a particular set of adversarial capabilities. Our empirical exploration validates our theoretical insights, where we can consistently reach 0% Attack Success Rate after a few adversarial retraining iterations.



## **3. Evaluations of Machine Learning Privacy Defenses are Misleading**

cs.CR

Accepted at ACM CCS 2024

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2404.17399v2) [paper-pdf](http://arxiv.org/pdf/2404.17399v2)

**Authors**: Michael Aerni, Jie Zhang, Florian Tramèr

**Abstract**: Empirical defenses for machine learning privacy forgo the provable guarantees of differential privacy in the hope of achieving higher utility while resisting realistic adversaries. We identify severe pitfalls in existing empirical privacy evaluations (based on membership inference attacks) that result in misleading conclusions. In particular, we show that prior evaluations fail to characterize the privacy leakage of the most vulnerable samples, use weak attacks, and avoid comparisons with practical differential privacy baselines. In 5 case studies of empirical privacy defenses, we find that prior evaluations underestimate privacy leakage by an order of magnitude. Under our stronger evaluation, none of the empirical defenses we study are competitive with a properly tuned, high-utility DP-SGD baseline (with vacuous provable guarantees).



## **4. Limited but consistent gains in adversarial robustness by co-training object recognition models with human EEG**

cs.LG

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03646v1) [paper-pdf](http://arxiv.org/pdf/2409.03646v1)

**Authors**: Manshan Guo, Bhavin Choksi, Sari Sadiya, Alessandro T. Gifford, Martina G. Vilas, Radoslaw M. Cichy, Gemma Roig

**Abstract**: In contrast to human vision, artificial neural networks (ANNs) remain relatively susceptible to adversarial attacks. To address this vulnerability, efforts have been made to transfer inductive bias from human brains to ANNs, often by training the ANN representations to match their biological counterparts. Previous works relied on brain data acquired in rodents or primates using invasive techniques, from specific regions of the brain, under non-natural conditions (anesthetized animals), and with stimulus datasets lacking diversity and naturalness. In this work, we explored whether aligning model representations to human EEG responses to a rich set of real-world images increases robustness to ANNs. Specifically, we trained ResNet50-backbone models on a dual task of classification and EEG prediction; and evaluated their EEG prediction accuracy and robustness to adversarial attacks. We observed significant correlation between the networks' EEG prediction accuracy, often highest around 100 ms post stimulus onset, and their gains in adversarial robustness. Although effect size was limited, effects were consistent across different random initializations and robust for architectural variants. We further teased apart the data from individual EEG channels and observed strongest contribution from electrodes in the parieto-occipital regions. The demonstrated utility of human EEG for such tasks opens up avenues for future efforts that scale to larger datasets under diverse stimuli conditions with the promise of stronger effects.



## **5. A practical approach to evaluating the adversarial distance for machine learning classifiers**

cs.LG

Accepted manuscript at International Mechanical Engineering Congress  and Exposition IMECE2024

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03598v1) [paper-pdf](http://arxiv.org/pdf/2409.03598v1)

**Authors**: Georg Siedel, Ekagra Gupta, Andrey Morozov

**Abstract**: Robustness is critical for machine learning (ML) classifiers to ensure consistent performance in real-world applications where models may encounter corrupted or adversarial inputs. In particular, assessing the robustness of classifiers to adversarial inputs is essential to protect systems from vulnerabilities and thus ensure safety in use. However, methods to accurately compute adversarial robustness have been challenging for complex ML models and high-dimensional data. Furthermore, evaluations typically measure adversarial accuracy on specific attack budgets, limiting the informative value of the resulting metrics. This paper investigates the estimation of the more informative adversarial distance using iterative adversarial attacks and a certification approach. Combined, the methods provide a comprehensive evaluation of adversarial robustness by computing estimates for the upper and lower bounds of the adversarial distance. We present visualisations and ablation studies that provide insights into how this evaluation method should be applied and parameterised. We find that our adversarial attack approach is effective compared to related implementations, while the certification method falls short of expectations. The approach in this paper should encourage a more informative way of evaluating the adversarial robustness of ML classifiers.



## **6. Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review**

cs.CL

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2310.14735v5) [paper-pdf](http://arxiv.org/pdf/2310.14735v5)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). The development of Artificial Intelligence (AI), from its inception in the 1950s to the emergence of advanced neural networks and deep learning architectures, has made a breakthrough in LLMs, with models such as GPT-4o and Claude-3, and in Vision-Language Models (VLMs), with models such as CLIP and ALIGN. Prompt engineering is the process of structuring inputs, which has emerged as a crucial technique to maximize the utility and accuracy of these models. This paper explores both foundational and advanced methodologies of prompt engineering, including techniques such as self-consistency, chain-of-thought, and generated knowledge, which significantly enhance model performance. Additionally, it examines the prompt method of VLMs through innovative approaches such as Context Optimization (CoOp), Conditional Context Optimization (CoCoOp), and Multimodal Prompt Learning (MaPLe). Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is also addressed, through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review also reflects the essential role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.



## **7. TSFool: Crafting Highly-Imperceptible Adversarial Time Series through Multi-Objective Attack**

cs.LG

27th European Conference on Artificial Intelligence (ECAI'24)

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2209.06388v4) [paper-pdf](http://arxiv.org/pdf/2209.06388v4)

**Authors**: Yanyun Wang, Dehui Du, Haibo Hu, Zi Liang, Yuanhao Liu

**Abstract**: Recent years have witnessed the success of recurrent neural network (RNN) models in time series classification (TSC). However, neural networks (NNs) are vulnerable to adversarial samples, which cause real-life adversarial attacks that undermine the robustness of AI models. To date, most existing attacks target at feed-forward NNs and image recognition tasks, but they cannot perform well on RNN-based TSC. This is due to the cyclical computation of RNN, which prevents direct model differentiation. In addition, the high visual sensitivity of time series to perturbations also poses challenges to local objective optimization of adversarial samples. In this paper, we propose an efficient method called TSFool to craft highly-imperceptible adversarial time series for RNN-based TSC. The core idea is a new global optimization objective known as "Camouflage Coefficient" that captures the imperceptibility of adversarial samples from the class distribution. Based on this, we reduce the adversarial attack problem to a multi-objective optimization problem that enhances the perturbation quality. Furthermore, to speed up the optimization process, we propose to use a representation model for RNN to capture deeply embedded vulnerable samples whose features deviate from the latent manifold. Experiments on 11 UCR and UEA datasets showcase that TSFool significantly outperforms six white-box and three black-box benchmark attacks in terms of effectiveness, efficiency and imperceptibility from various perspectives including standard measure, human study and real-world defense.



## **8. Boosting Adversarial Transferability for Skeleton-based Action Recognition via Exploring the Model Posterior Space**

cs.CV

We have submitted a new version of our work at arXiv:2409.02483. This  version, arXiv:2407.08572, is no longer valid. Any update for this work will  be conducted in arXiv:2409.02483

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2407.08572v2) [paper-pdf](http://arxiv.org/pdf/2407.08572v2)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Xun Yang, Meng Wang, He Wang

**Abstract**: Skeletal motion plays a pivotal role in human activity recognition (HAR). Recently, attack methods have been proposed to identify the universal vulnerability of skeleton-based HAR(S-HAR). However, the research of adversarial transferability on S-HAR is largely missing. More importantly, existing attacks all struggle in transfer across unknown S-HAR models. We observed that the key reason is that the loss landscape of the action recognizers is rugged and sharp. Given the established correlation in prior studies~\cite{qin2022boosting,wu2020towards} between loss landscape and adversarial transferability, we assume and empirically validate that smoothing the loss landscape could potentially improve adversarial transferability on S-HAR. This is achieved by proposing a new post-train Dual Bayesian strategy, which can effectively explore the model posterior space for a collection of surrogates without the need for re-training. Furthermore, to craft adversarial examples along the motion manifold, we incorporate the attack gradient with information of the motion dynamics in a Bayesian manner. Evaluated on benchmark datasets, e.g. HDM05 and NTU 60, the average transfer success rate can reach as high as 35.9\% and 45.5\% respectively. In comparison, current state-of-the-art skeletal attacks achieve only 3.6\% and 9.8\%. The high adversarial transferability remains consistent across various surrogate, victim, and even defense models. Through a comprehensive analysis of the results, we provide insights on what surrogates are more likely to exhibit transferability, to shed light on future research.



## **9. LLM Detectors Still Fall Short of Real World: Case of LLM-Generated Short News-Like Posts**

cs.CL

20 pages, 7 tables, 13 figures, under consideration for EMNLP

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03291v1) [paper-pdf](http://arxiv.org/pdf/2409.03291v1)

**Authors**: Henrique Da Silva Gameiro, Andrei Kucharavy, Ljiljana Dolamic

**Abstract**: With the emergence of widely available powerful LLMs, disinformation generated by large Language Models (LLMs) has become a major concern. Historically, LLM detectors have been touted as a solution, but their effectiveness in the real world is still to be proven. In this paper, we focus on an important setting in information operations -- short news-like posts generated by moderately sophisticated attackers.   We demonstrate that existing LLM detectors, whether zero-shot or purpose-trained, are not ready for real-world use in that setting. All tested zero-shot detectors perform inconsistently with prior benchmarks and are highly vulnerable to sampling temperature increase, a trivial attack absent from recent benchmarks. A purpose-trained detector generalizing across LLMs and unseen attacks can be developed, but it fails to generalize to new human-written texts.   We argue that the former indicates domain-specific benchmarking is needed, while the latter suggests a trade-off between the adversarial evasion resilience and overfitting to the reference human text, with both needing evaluation in benchmarks and currently absent. We believe this suggests a re-consideration of current LLM detector benchmarking approaches and provides a dynamically extensible benchmark to allow it (https://github.com/Reliable-Information-Lab-HEVS/dynamic_llm_detector_benchmark).



## **10. OpenFact at CheckThat! 2024: Combining Multiple Attack Methods for Effective Adversarial Text Generation**

cs.CL

CLEF 2024 - Conference and Labs of the Evaluation Forum

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.02649v2) [paper-pdf](http://arxiv.org/pdf/2409.02649v2)

**Authors**: Włodzimierz Lewoniewski, Piotr Stolarski, Milena Stróżyna, Elzbieta Lewańska, Aleksandra Wojewoda, Ewelina Księżniak, Marcin Sawiński

**Abstract**: This paper presents the experiments and results for the CheckThat! Lab at CLEF 2024 Task 6: Robustness of Credibility Assessment with Adversarial Examples (InCrediblAE). The primary objective of this task was to generate adversarial examples in five problem domains in order to evaluate the robustness of widely used text classification methods (fine-tuned BERT, BiLSTM, and RoBERTa) when applied to credibility assessment issues.   This study explores the application of ensemble learning to enhance adversarial attacks on natural language processing (NLP) models. We systematically tested and refined several adversarial attack methods, including BERT-Attack, Genetic algorithms, TextFooler, and CLARE, on five datasets across various misinformation tasks. By developing modified versions of BERT-Attack and hybrid methods, we achieved significant improvements in attack effectiveness. Our results demonstrate the potential of modification and combining multiple methods to create more sophisticated and effective adversarial attack strategies, contributing to the development of more robust and secure systems.



## **11. AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization**

cs.CV

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2402.11940v3) [paper-pdf](http://arxiv.org/pdf/2402.11940v3)

**Authors**: Jiyao Li, Mingze Ni, Yifei Dong, Tianqing Zhu, Wei Liu

**Abstract**: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. This paper presents a novel adversarial attack strategy, AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by a customised differential evolution method to optimise the perturbations of pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchmark datasets against multiple victim models. The experimental results demonstrate that our method outperforms current leading-edge techniques by achieving consistently higher attack success rates.



## **12. Robust Q-Learning under Corrupted Rewards**

cs.LG

Accepted to the Decision and Control Conference (CDC) 2024

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03237v1) [paper-pdf](http://arxiv.org/pdf/2409.03237v1)

**Authors**: Sreejeet Maity, Aritra Mitra

**Abstract**: Recently, there has been a surge of interest in analyzing the non-asymptotic behavior of model-free reinforcement learning algorithms. However, the performance of such algorithms in non-ideal environments, such as in the presence of corrupted rewards, is poorly understood. Motivated by this gap, we investigate the robustness of the celebrated Q-learning algorithm to a strong-contamination attack model, where an adversary can arbitrarily perturb a small fraction of the observed rewards. We start by proving that such an attack can cause the vanilla Q-learning algorithm to incur arbitrarily large errors. We then develop a novel robust synchronous Q-learning algorithm that uses historical reward data to construct robust empirical Bellman operators at each time step. Finally, we prove a finite-time convergence rate for our algorithm that matches known state-of-the-art bounds (in the absence of attacks) up to a small inevitable $O(\varepsilon)$ error term that scales with the adversarial corruption fraction $\varepsilon$. Notably, our results continue to hold even when the true reward distributions have infinite support, provided they admit bounded second moments.



## **13. Transfer-based Adversarial Poisoning Attacks for Online (MIMO-)Deep Receviers**

eess.SP

15 pages, 14 figures

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.02430v2) [paper-pdf](http://arxiv.org/pdf/2409.02430v2)

**Authors**: Kunze Wu, Weiheng Jiang, Dusit Niyato, Yinghuan Li, Chuang Luo

**Abstract**: Recently, the design of wireless receivers using deep neural networks (DNNs), known as deep receivers, has attracted extensive attention for ensuring reliable communication in complex channel environments. To adapt quickly to dynamic channels, online learning has been adopted to update the weights of deep receivers with over-the-air data (e.g., pilots). However, the fragility of neural models and the openness of wireless channels expose these systems to malicious attacks. To this end, understanding these attack methods is essential for robust receiver design. In this paper, we propose a transfer-based adversarial poisoning attack method for online receivers.Without knowledge of the attack target, adversarial perturbations are injected to the pilots, poisoning the online deep receiver and impairing its ability to adapt to dynamic channels and nonlinear effects. In particular, our attack method targets Deep Soft Interference Cancellation (DeepSIC)[1] using online meta-learning. As a classical model-driven deep receiver, DeepSIC incorporates wireless domain knowledge into its architecture. This integration allows it to adapt efficiently to time-varying channels with only a small number of pilots, achieving optimal performance in a multi-input and multi-output (MIMO) scenario.The deep receiver in this scenario has a number of applications in the field of wireless communication, which motivates our study of the attack methods targeting it.Specifically, we demonstrate the effectiveness of our attack in simulations on synthetic linear, synthetic nonlinear, static, and COST 2100 channels. Simulation results indicate that the proposed poisoning attack significantly reduces the performance of online receivers in rapidly changing scenarios.



## **14. Bypassing DARCY Defense: Indistinguishable Universal Adversarial Triggers**

cs.CL

13 pages, 5 figures

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03183v1) [paper-pdf](http://arxiv.org/pdf/2409.03183v1)

**Authors**: Zuquan Peng, Yuanyuan He, Jianbing Ni, Ben Niu

**Abstract**: Neural networks (NN) classification models for Natural Language Processing (NLP) are vulnerable to the Universal Adversarial Triggers (UAT) attack that triggers a model to produce a specific prediction for any input. DARCY borrows the "honeypot" concept to bait multiple trapdoors, effectively detecting the adversarial examples generated by UAT. Unfortunately, we find a new UAT generation method, called IndisUAT, which produces triggers (i.e., tokens) and uses them to craft adversarial examples whose feature distribution is indistinguishable from that of the benign examples in a randomly-chosen category at the detection layer of DARCY. The produced adversarial examples incur the maximal loss of predicting results in the DARCY-protected models. Meanwhile, the produced triggers are effective in black-box models for text generation, text inference, and reading comprehension. Finally, the evaluation results under NN models for NLP tasks indicate that the IndisUAT method can effectively circumvent DARCY and penetrate other defenses. For example, IndisUAT can reduce the true positive rate of DARCY's detection by at least 40.8% and 90.6%, and drop the accuracy by at least 33.3% and 51.6% in the RNN and CNN models, respectively. IndisUAT reduces the accuracy of the BERT's adversarial defense model by at least 34.0%, and makes the GPT-2 language model spew racist outputs even when conditioned on non-racial context.



## **15. ACCESS-FL: Agile Communication and Computation for Efficient Secure Aggregation in Stable Federated Learning Networks**

cs.CR

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.01722v2) [paper-pdf](http://arxiv.org/pdf/2409.01722v2)

**Authors**: Niousha Nazemi, Omid Tavallaie, Shuaijun Chen, Anna Maria Mandalari, Kanchana Thilakarathna, Ralph Holz, Hamed Haddadi, Albert Y. Zomaya

**Abstract**: Federated Learning (FL) is a promising distributed learning framework designed for privacy-aware applications. FL trains models on client devices without sharing the client's data and generates a global model on a server by aggregating model updates. Traditional FL approaches risk exposing sensitive client data when plain model updates are transmitted to the server, making them vulnerable to security threats such as model inversion attacks where the server can infer the client's original training data from monitoring the changes of the trained model in different rounds. Google's Secure Aggregation (SecAgg) protocol addresses this threat by employing a double-masking technique, secret sharing, and cryptography computations in honest-but-curious and adversarial scenarios with client dropouts. However, in scenarios without the presence of an active adversary, the computational and communication cost of SecAgg significantly increases by growing the number of clients. To address this issue, in this paper, we propose ACCESS-FL, a communication-and-computation-efficient secure aggregation method designed for honest-but-curious scenarios in stable FL networks with a limited rate of client dropout. ACCESS-FL reduces the computation/communication cost to a constant level (independent of the network size) by generating shared secrets between only two clients and eliminating the need for double masking, secret sharing, and cryptography computations. To evaluate the performance of ACCESS-FL, we conduct experiments using the MNIST, FMNIST, and CIFAR datasets to verify the performance of our proposed method. The evaluation results demonstrate that our proposed method significantly reduces computation and communication overhead compared to state-of-the-art methods, SecAgg and SecAgg+.



## **16. Well, that escalated quickly: The Single-Turn Crescendo Attack (STCA)**

cs.CR

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.03131v1) [paper-pdf](http://arxiv.org/pdf/2409.03131v1)

**Authors**: Alan Aqrawi

**Abstract**: This paper explores a novel approach to adversarial attacks on large language models (LLM): the Single-Turn Crescendo Attack (STCA). The STCA builds upon the multi-turn crescendo attack established by Mark Russinovich, Ahmed Salem, Ronen Eldan. Traditional multi-turn adversarial strategies gradually escalate the context to elicit harmful or controversial responses from LLMs. However, this paper introduces a more efficient method where the escalation is condensed into a single interaction. By carefully crafting the prompt to simulate an extended dialogue, the attack bypasses typical content moderation systems, leading to the generation of responses that would normally be filtered out. I demonstrate this technique through a few case studies. The results highlight vulnerabilities in current LLMs and underscore the need for more robust safeguards. This work contributes to the broader discourse on responsible AI (RAI) safety and adversarial testing, providing insights and practical examples for researchers and developers. This method is unexplored in the literature, making it a novel contribution to the field.



## **17. Knowledge Transfer for Collaborative Misbehavior Detection in Untrusted Vehicular Environments**

cs.NI

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02844v1) [paper-pdf](http://arxiv.org/pdf/2409.02844v1)

**Authors**: Roshan Sedar, Charalampos Kalalas, Paolo Dini, Francisco Vazquez-Gallego, Jesus Alonso-Zarate, Luis Alonso

**Abstract**: Vehicular mobility underscores the need for collaborative misbehavior detection at the vehicular edge. However, locally trained misbehavior detection models are susceptible to adversarial attacks that aim to deliberately influence learning outcomes. In this paper, we introduce a deep reinforcement learning-based approach that employs transfer learning for collaborative misbehavior detection among roadside units (RSUs). In the presence of label-flipping and policy induction attacks, we perform selective knowledge transfer from trustworthy source RSUs to foster relevant expertise in misbehavior detection and avoid negative knowledge sharing from adversary-influenced RSUs. The performance of our proposed scheme is demonstrated with evaluations over a diverse set of misbehavior detection scenarios using an open-source dataset. Experimental results show that our approach significantly reduces the training time at the target RSU and achieves superior detection performance compared to the baseline scheme with tabula rasa learning. Enhanced robustness and generalizability can also be attained, by effectively detecting previously unseen and partially observable misbehavior attacks.



## **18. Revisiting Character-level Adversarial Attacks for Language Models**

cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2405.04346v2) [paper-pdf](http://arxiv.org/pdf/2405.04346v2)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.



## **19. Simple fusion-fission quantifies Israel-Palestine violence and suggests multi-adversary solution**

physics.soc-ph

Comments welcome. Working paper

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02816v1) [paper-pdf](http://arxiv.org/pdf/2409.02816v1)

**Authors**: Frank Yingjie Huo, Pedro D. Manrique, Dylan J. Restrepo, Gordon Woo, Neil F. Johnson

**Abstract**: Why humans fight has no easy answer. However, understanding better how humans fight could inform future interventions, hidden shifts and casualty risk. Fusion-fission describes the well-known grouping behavior of fish etc. fighting for survival in the face of strong opponents: they form clusters ('fusion') which provide collective benefits and a cluster scatters when it senses danger ('fission'). Here we show how similar clustering (fusion-fission) of human fighters provides a unified quantitative explanation for complex casualty patterns across decades of Israel-Palestine region violence, as well as the October 7 surprise attack -- and uncovers a hidden post-October 7 shift. State-of-the-art data shows this fighter fusion-fission in action. It also predicts future 'super-shock' attacks that will be more lethal than October 7 and will arrive earlier. It offers a multi-adversary solution. Our results -- which include testable formulae and a plug-and-play simulation -- enable concrete risk assessments of future casualties and policy-making grounded by fighter behavior.



## **20. Boosting Certificate Robustness for Time Series Classification with Efficient Self-Ensemble**

cs.LG

6 figures, 4 tables, 10 pages

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02802v1) [paper-pdf](http://arxiv.org/pdf/2409.02802v1)

**Authors**: Chang Dong, Zhengyang Li, Liangwei Zheng, Weitong Chen, Wei Emma Zhang

**Abstract**: Recently, the issue of adversarial robustness in the time series domain has garnered significant attention. However, the available defense mechanisms remain limited, with adversarial training being the predominant approach, though it does not provide theoretical guarantees. Randomized Smoothing has emerged as a standout method due to its ability to certify a provable lower bound on robustness radius under $\ell_p$-ball attacks. Recognizing its success, research in the time series domain has started focusing on these aspects. However, existing research predominantly focuses on time series forecasting, or under the non-$\ell_p$ robustness in statistic feature augmentation for time series classification~(TSC). Our review found that Randomized Smoothing performs modestly in TSC, struggling to provide effective assurances on datasets with poor robustness. Therefore, we propose a self-ensemble method to enhance the lower bound of the probability confidence of predicted labels by reducing the variance of classification margins, thereby certifying a larger radius. This approach also addresses the computational overhead issue of Deep Ensemble~(DE) while remaining competitive and, in some cases, outperforming it in terms of robustness. Both theoretical analysis and experimental results validate the effectiveness of our method, demonstrating superior performance in robustness testing compared to baseline approaches.



## **21. AdvSecureNet: A Python Toolkit for Adversarial Machine Learning**

cs.CV

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02629v1) [paper-pdf](http://arxiv.org/pdf/2409.02629v1)

**Authors**: Melih Catal, Manuel Günther

**Abstract**: Machine learning models are vulnerable to adversarial attacks. Several tools have been developed to research these vulnerabilities, but they often lack comprehensive features and flexibility. We introduce AdvSecureNet, a PyTorch based toolkit for adversarial machine learning that is the first to natively support multi-GPU setups for attacks, defenses, and evaluation. It is the first toolkit that supports both CLI and API interfaces and external YAML configuration files to enhance versatility and reproducibility. The toolkit includes multiple attacks, defenses and evaluation metrics. Rigiorous software engineering practices are followed to ensure high code quality and maintainability. The project is available as an open-source project on GitHub at https://github.com/melihcatal/advsecurenet and installable via PyPI.



## **22. Adversarial Attacks on Machine Learning-Aided Visualizations**

cs.CR

This is the author's version of the article that has been accepted by  the Journal of Visualization

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02485v1) [paper-pdf](http://arxiv.org/pdf/2409.02485v1)

**Authors**: Takanori Fujiwara, Kostiantyn Kucher, Junpeng Wang, Rafael M. Martins, Andreas Kerren, Anders Ynnerman

**Abstract**: Research in ML4VIS investigates how to use machine learning (ML) techniques to generate visualizations, and the field is rapidly growing with high societal impact. However, as with any computational pipeline that employs ML processes, ML4VIS approaches are susceptible to a range of ML-specific adversarial attacks. These attacks can manipulate visualization generations, causing analysts to be tricked and their judgments to be impaired. Due to a lack of synthesis from both visualization and ML perspectives, this security aspect is largely overlooked by the current ML4VIS literature. To bridge this gap, we investigate the potential vulnerabilities of ML-aided visualizations from adversarial attacks using a holistic lens of both visualization and ML perspectives. We first identify the attack surface (i.e., attack entry points) that is unique in ML-aided visualizations. We then exemplify five different adversarial attacks. These examples highlight the range of possible attacks when considering the attack surface and multiple different adversary capabilities. Our results show that adversaries can induce various attacks, such as creating arbitrary and deceptive visualizations, by systematically identifying input attributes that are influential in ML inferences. Based on our observations of the attack surface characteristics and the attack examples, we underline the importance of comprehensive studies of security issues and defense mechanisms as a call of urgency for the ML4VIS community.



## **23. TASAR: Transferable Attack on Skeletal Action Recognition**

cs.CV

arXiv admin note: text overlap with arXiv:2407.08572

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02483v1) [paper-pdf](http://arxiv.org/pdf/2409.02483v1)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Ajian Liu, Xingxing Wei, Meng Wang, He Wang

**Abstract**: Skeletal sequences, as well-structured representations of human behaviors, are crucial in Human Activity Recognition (HAR). The transferability of adversarial skeletal sequences enables attacks in real-world HAR scenarios, such as autonomous driving, intelligent surveillance, and human-computer interactions. However, existing Skeleton-based HAR (S-HAR) attacks exhibit weak adversarial transferability and, therefore, cannot be considered true transfer-based S-HAR attacks. More importantly, the reason for this failure remains unclear. In this paper, we study this phenomenon through the lens of loss surface, and find that its sharpness contributes to the poor transferability in S-HAR. Inspired by this observation, we assume and empirically validate that smoothening the rugged loss landscape could potentially improve adversarial transferability in S-HAR. To this end, we propose the first Transfer-based Attack on Skeletal Action Recognition, TASAR. TASAR explores the smoothed model posterior without re-training the pre-trained surrogates, which is achieved by a new post-train Dual Bayesian optimization strategy. Furthermore, unlike previous transfer-based attacks that treat each frame independently and overlook temporal coherence within sequences, TASAR incorporates motion dynamics into the Bayesian attack gradient, effectively disrupting the spatial-temporal coherence of S-HARs. To exhaustively evaluate the effectiveness of existing methods and our method, we build the first large-scale robust S-HAR benchmark, comprising 7 S-HAR models, 10 attack methods, 3 S-HAR datasets and 2 defense models. Extensive results demonstrate the superiority of TASAR. Our benchmark enables easy comparisons for future studies, with the code available in the supplementary material.



## **24. Jailbreaking Prompt Attack: A Controllable Adversarial Attack against Diffusion Models**

cs.CR

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2404.02928v3) [paper-pdf](http://arxiv.org/pdf/2404.02928v3)

**Authors**: Jiachen Ma, Anda Cao, Zhiqing Xiao, Yijiang Li, Jie Zhang, Chao Ye, Junbo Zhao

**Abstract**: Text-to-image (T2I) models can be maliciously used to generate harmful content such as sexually explicit, unfaithful, and misleading or Not-Safe-for-Work (NSFW) images. Previous attacks largely depend on the availability of the diffusion model or involve a lengthy optimization process. In this work, we investigate a more practical and universal attack that does not require the presence of a target model and demonstrate that the high-dimensional text embedding space inherently contains NSFW concepts that can be exploited to generate harmful images. We present the Jailbreaking Prompt Attack (JPA). JPA first searches for the target malicious concepts in the text embedding space using a group of antonyms generated by ChatGPT. Subsequently, a prefix prompt is optimized in the discrete vocabulary space to align malicious concepts semantically in the text embedding space. We further introduce a soft assignment with gradient masking technique that allows us to perform gradient ascent in the discrete vocabulary space.   We perform extensive experiments with open-sourced T2I models, e.g. stable-diffusion-v1-4 and closed-sourced online services, e.g. DALLE2, Midjourney with black-box safety checkers. Results show that (1) JPA bypasses both text and image safety checkers (2) while preserving high semantic alignment with the target prompt. (3) JPA demonstrates a much faster speed than previous methods and can be executed in a fully automated manner. These merits render it a valuable tool for robustness evaluation in future text-to-image generation research.



## **25. LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet**

cs.LG

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2408.15221v2) [paper-pdf](http://arxiv.org/pdf/2408.15221v2)

**Authors**: Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, Summer Yue

**Abstract**: Recent large language model (LLM) defenses have greatly improved models' ability to refuse harmful queries, even when adversarially attacked. However, LLM defenses are primarily evaluated against automated adversarial attacks in a single turn of conversation, an insufficient threat model for real-world malicious use. We demonstrate that multi-turn human jailbreaks uncover significant vulnerabilities, exceeding 70% attack success rate (ASR) on HarmBench against defenses that report single-digit ASRs with automated single-turn attacks. Human jailbreaks also reveal vulnerabilities in machine unlearning defenses, successfully recovering dual-use biosecurity knowledge from unlearned models. We compile these results into Multi-Turn Human Jailbreaks (MHJ), a dataset of 2,912 prompts across 537 multi-turn jailbreaks. We publicly release MHJ alongside a compendium of jailbreak tactics developed across dozens of commercial red teaming engagements, supporting research towards stronger LLM defenses.



## **26. RAMBO: Leaking Secrets from Air-Gap Computers by Spelling Covert Radio Signals from Computer RAM**

cs.CR

Version of this work accepted to Nordic Conference on Secure IT  Systems, 2023

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.02292v1) [paper-pdf](http://arxiv.org/pdf/2409.02292v1)

**Authors**: Mordechai Guri

**Abstract**: Air-gapped systems are physically separated from external networks, including the Internet. This isolation is achieved by keeping the air-gap computers disconnected from wired or wireless networks, preventing direct or remote communication with other devices or networks. Air-gap measures may be used in sensitive environments where security and isolation are critical to prevent private and confidential information leakage.   In this paper, we present an attack allowing adversaries to leak information from air-gapped computers. We show that malware on a compromised computer can generate radio signals from memory buses (RAM). Using software-generated radio signals, malware can encode sensitive information such as files, images, keylogging, biometric information, and encryption keys. With software-defined radio (SDR) hardware, and a simple off-the-shelf antenna, an attacker can intercept transmitted raw radio signals from a distance. The signals can then be decoded and translated back into binary information. We discuss the design and implementation and present related work and evaluation results. This paper presents fast modification methods to leak data from air-gapped computers at 1000 bits per second. Finally, we propose countermeasures to mitigate this out-of-band air-gap threat.



## **27. NoiseAttack: An Evasive Sample-Specific Multi-Targeted Backdoor Attack Through White Gaussian Noise**

cs.CV

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.02251v1) [paper-pdf](http://arxiv.org/pdf/2409.02251v1)

**Authors**: Abdullah Arafat Miah, Kaan Icer, Resit Sendag, Yu Bi

**Abstract**: Backdoor attacks pose a significant threat when using third-party data for deep learning development. In these attacks, data can be manipulated to cause a trained model to behave improperly when a specific trigger pattern is applied, providing the adversary with unauthorized advantages. While most existing works focus on designing trigger patterns in both visible and invisible to poison the victim class, they typically result in a single targeted class upon the success of the backdoor attack, meaning that the victim class can only be converted to another class based on the adversary predefined value. In this paper, we address this issue by introducing a novel sample-specific multi-targeted backdoor attack, namely NoiseAttack. Specifically, we adopt White Gaussian Noise (WGN) with various Power Spectral Densities (PSD) as our underlying triggers, coupled with a unique training strategy to execute the backdoor attack. This work is the first of its kind to launch a vision backdoor attack with the intent to generate multiple targeted classes with minimal input configuration. Furthermore, our extensive experimental results demonstrate that NoiseAttack can achieve a high attack success rate against popular network architectures and datasets, as well as bypass state-of-the-art backdoor detection methods. Our source code and experiments are available at https://github.com/SiSL-URI/NoiseAttack/tree/main.



## **28. Quantifying Liveness and Safety of Avalanche's Snowball**

cs.DC

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.02217v1) [paper-pdf](http://arxiv.org/pdf/2409.02217v1)

**Authors**: Quentin Kniep, Maxime Laval, Jakub Sliwinski, Roger Wattenhofer

**Abstract**: This work examines the resilience properties of the Snowball and Avalanche protocols that underlie the popular Avalanche blockchain. We experimentally quantify the resilience of Snowball using a simulation implemented in Rust, where the adversary strategically rebalances the network to delay termination.   We show that in a network of $n$ nodes of equal stake, the adversary is able to break liveness when controlling $\Omega(\sqrt{n})$ nodes. Specifically, for $n = 2000$, a simple adversary controlling $5.2\%$ of stake can successfully attack liveness. When the adversary is given additional information about the state of the network (without any communication or other advantages), the stake needed for a successful attack is as little as $2.8\%$. We show that the adversary can break safety in time exponentially dependent on their stake, and inversely linearly related to the size of the network, e.g. in 265 rounds in expectation when the adversary controls $25\%$ of a network of 3000.   We conclude that Snowball and Avalanche are akin to Byzantine reliable broadcast protocols as opposed to consensus.



## **29. Learning Resilient Formation Control of Drones with Graph Attention Network**

cs.RO

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01953v1) [paper-pdf](http://arxiv.org/pdf/2409.01953v1)

**Authors**: Jiaping Xiao, Xu Fang, Qianlei Jia, Mir Feroskhan

**Abstract**: The rapid advancement of drone technology has significantly impacted various sectors, including search and rescue, environmental surveillance, and industrial inspection. Multidrone systems offer notable advantages such as enhanced efficiency, scalability, and redundancy over single-drone operations. Despite these benefits, ensuring resilient formation control in dynamic and adversarial environments, such as under communication loss or cyberattacks, remains a significant challenge. Classical approaches to resilient formation control, while effective in certain scenarios, often struggle with complex modeling and the curse of dimensionality, particularly as the number of agents increases. This paper proposes a novel, learning-based formation control for enhancing the adaptability and resilience of multidrone formations using graph attention networks (GATs). By leveraging GAT's dynamic capabilities to extract internode relationships based on the attention mechanism, this GAT-based formation controller significantly improves the robustness of drone formations against various threats, such as Denial of Service (DoS) attacks. Our approach not only improves formation performance in normal conditions but also ensures the resilience of multidrone systems in variable and adversarial environments. Extensive simulation results demonstrate the superior performance of our method over baseline formation controllers. Furthermore, the physical experiments validate the effectiveness of the trained control policy in real-world flights.



## **30. Reassessing Noise Augmentation Methods in the Context of Adversarial Speech**

eess.AS

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01813v1) [paper-pdf](http://arxiv.org/pdf/2409.01813v1)

**Authors**: Karla Pizzi, Matías P. Pizarro B, Asja Fischer

**Abstract**: In this study, we investigate if noise-augmented training can concurrently improve adversarial robustness in automatic speech recognition (ASR) systems. We conduct a comparative analysis of the adversarial robustness of four different state-of-the-art ASR architectures, where each of the ASR architectures is trained under three different augmentation conditions: one subject to background noise, speed variations, and reverberations, another subject to speed variations only, and a third without any form of data augmentation. The results demonstrate that noise augmentation not only improves model performance on noisy speech but also the model's robustness to adversarial attacks.



## **31. USTC-KXDIGIT System Description for ASVspoof5 Challenge**

cs.SD

ASVspoof5 workshop paper

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01695v1) [paper-pdf](http://arxiv.org/pdf/2409.01695v1)

**Authors**: Yihao Chen, Haochen Wu, Nan Jiang, Xiang Xia, Qing Gu, Yunqi Hao, Pengfei Cai, Yu Guan, Jialong Wang, Weilin Xie, Lei Fang, Sian Fang, Yan Song, Wu Guo, Lin Liu, Minqiang Xu

**Abstract**: This paper describes the USTC-KXDIGIT system submitted to the ASVspoof5 Challenge for Track 1 (speech deepfake detection) and Track 2 (spoofing-robust automatic speaker verification, SASV). Track 1 showcases a diverse range of technical qualities from potential processing algorithms and includes both open and closed conditions. For these conditions, our system consists of a cascade of a frontend feature extractor and a back-end classifier. We focus on extensive embedding engineering and enhancing the generalization of the back-end classifier model. Specifically, the embedding engineering is based on hand-crafted features and speech representations from a self-supervised model, used for closed and open conditions, respectively. To detect spoof attacks under various adversarial conditions, we trained multiple systems on an augmented training set. Additionally, we used voice conversion technology to synthesize fake audio from genuine audio in the training set to enrich the synthesis algorithms. To leverage the complementary information learned by different model architectures, we employed activation ensemble and fused scores from different systems to obtain the final decision score for spoof detection. During the evaluation phase, the proposed methods achieved 0.3948 minDCF and 14.33% EER in the close condition, and 0.0750 minDCF and 2.59% EER in the open condition, demonstrating the robustness of our submitted systems under adversarial conditions. In Track 2, we continued using the CM system from Track 1 and fused it with a CNN-based ASV system. This approach achieved 0.2814 min-aDCF in the closed condition and 0.0756 min-aDCF in the open condition, showcasing superior performance in the SASV system.



## **32. Purification-Agnostic Proxy Learning for Agentic Copyright Watermarking against Adversarial Evidence Forgery**

cs.CV

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01541v1) [paper-pdf](http://arxiv.org/pdf/2409.01541v1)

**Authors**: Erjin Bao, Ching-Chun Chang, Hanrui Wang, Isao Echizen

**Abstract**: With the proliferation of AI agents in various domains, protecting the ownership of AI models has become crucial due to the significant investment in their development. Unauthorized use and illegal distribution of these models pose serious threats to intellectual property, necessitating effective copyright protection measures. Model watermarking has emerged as a key technique to address this issue, embedding ownership information within models to assert rightful ownership during copyright disputes. This paper presents several contributions to model watermarking: a self-authenticating black-box watermarking protocol using hash techniques, a study on evidence forgery attacks using adversarial perturbations, a proposed defense involving a purification step to counter adversarial attacks, and a purification-agnostic proxy learning method to enhance watermark reliability and model performance. Experimental results demonstrate the effectiveness of these approaches in improving the security, reliability, and performance of watermarked models.



## **33. On Evaluating Adversarial Robustness of Volumetric Medical Segmentation Models**

eess.IV

Accepted at British Machine Vision Conference 2024

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2406.08486v2) [paper-pdf](http://arxiv.org/pdf/2406.08486v2)

**Authors**: Hashmat Shadab Malik, Numan Saeed, Asif Hanif, Muzammal Naseer, Mohammad Yaqub, Salman Khan, Fahad Shahbaz Khan

**Abstract**: Volumetric medical segmentation models have achieved significant success on organ and tumor-based segmentation tasks in recent years. However, their vulnerability to adversarial attacks remains largely unexplored, raising serious concerns regarding the real-world deployment of tools employing such models in the healthcare sector. This underscores the importance of investigating the robustness of existing models. In this context, our work aims to empirically examine the adversarial robustness across current volumetric segmentation architectures, encompassing Convolutional, Transformer, and Mamba-based models. We extend this investigation across four volumetric segmentation datasets, evaluating robustness under both white box and black box adversarial attacks. Overall, we observe that while both pixel and frequency-based attacks perform reasonably well under \emph{white box} setting, the latter performs significantly better under transfer-based black box attacks. Across our experiments, we observe transformer-based models show higher robustness than convolution-based models with Mamba-based models being the most vulnerable. Additionally, we show that large-scale training of volumetric segmentation models improves the model's robustness against adversarial attacks. The code and robust models are available at https://github.com/HashmatShadab/Robustness-of-Volumetric-Medical-Segmentation-Models.



## **34. One-Index Vector Quantization Based Adversarial Attack on Image Classification**

cs.CV

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2409.01282v1) [paper-pdf](http://arxiv.org/pdf/2409.01282v1)

**Authors**: Haiju Fan, Xiaona Qin, Shuang Chen, Hubert P. H. Shum, Ming Li

**Abstract**: To improve storage and transmission, images are generally compressed. Vector quantization (VQ) is a popular compression method as it has a high compression ratio that suppresses other compression techniques. Despite this, existing adversarial attack methods on image classification are mostly performed in the pixel domain with few exceptions in the compressed domain, making them less applicable in real-world scenarios. In this paper, we propose a novel one-index attack method in the VQ domain to generate adversarial images by a differential evolution algorithm, successfully resulting in image misclassification in victim models. The one-index attack method modifies a single index in the compressed data stream so that the decompressed image is misclassified. It only needs to modify a single VQ index to realize an attack, which limits the number of perturbed indexes. The proposed method belongs to a semi-black-box attack, which is more in line with the actual attack scenario. We apply our method to attack three popular image classification models, i.e., Resnet, NIN, and VGG16. On average, 55.9% and 77.4% of the images in CIFAR-10 and Fashion MNIST, respectively, are successfully attacked, with a high level of misclassification confidence and a low level of image perturbation.



## **35. A Review of Image Retrieval Techniques: Data Augmentation and Adversarial Learning Approaches**

cs.CV

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2409.01219v1) [paper-pdf](http://arxiv.org/pdf/2409.01219v1)

**Authors**: Kim Jinwoo

**Abstract**: Image retrieval is a crucial research topic in computer vision, with broad application prospects ranging from online product searches to security surveillance systems. In recent years, the accuracy and efficiency of image retrieval have significantly improved due to advancements in deep learning. However, existing methods still face numerous challenges, particularly in handling large-scale datasets, cross-domain retrieval, and image perturbations that can arise from real-world conditions such as variations in lighting, occlusion, and viewpoint. Data augmentation techniques and adversarial learning methods have been widely applied in the field of image retrieval to address these challenges. Data augmentation enhances the model's generalization ability and robustness by generating more diverse training samples, simulating real-world variations, and reducing overfitting. Meanwhile, adversarial attacks and defenses introduce perturbations during training to improve the model's robustness against potential attacks, ensuring reliability in practical applications. This review comprehensively summarizes the latest research advancements in image retrieval, with a particular focus on the roles of data augmentation and adversarial learning techniques in enhancing retrieval performance. Future directions and potential challenges are also discussed.



## **36. BadMerging: Backdoor Attacks Against Model Merging**

cs.CR

To appear in ACM Conference on Computer and Communications Security  (CCS), 2024

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2408.07362v2) [paper-pdf](http://arxiv.org/pdf/2408.07362v2)

**Authors**: Jinghuai Zhang, Jianfeng Chi, Zheng Li, Kunlin Cai, Yang Zhang, Yuan Tian

**Abstract**: Fine-tuning pre-trained models for downstream tasks has led to a proliferation of open-sourced task-specific models. Recently, Model Merging (MM) has emerged as an effective approach to facilitate knowledge transfer among these independently fine-tuned models. MM directly combines multiple fine-tuned task-specific models into a merged model without additional training, and the resulting model shows enhanced capabilities in multiple tasks. Although MM provides great utility, it may come with security risks because an adversary can exploit MM to affect multiple downstream tasks. However, the security risks of MM have barely been studied. In this paper, we first find that MM, as a new learning paradigm, introduces unique challenges for existing backdoor attacks due to the merging process. To address these challenges, we introduce BadMerging, the first backdoor attack specifically designed for MM. Notably, BadMerging allows an adversary to compromise the entire merged model by contributing as few as one backdoored task-specific model. BadMerging comprises a two-stage attack mechanism and a novel feature-interpolation-based loss to enhance the robustness of embedded backdoors against the changes of different merging parameters. Considering that a merged model may incorporate tasks from different domains, BadMerging can jointly compromise the tasks provided by the adversary (on-task attack) and other contributors (off-task attack) and solve the corresponding unique challenges with novel attack designs. Extensive experiments show that BadMerging achieves remarkable attacks against various MM algorithms. Our ablation study demonstrates that the proposed attack designs can progressively contribute to the attack performance. Finally, we show that prior defense mechanisms fail to defend against our attacks, highlighting the need for more advanced defense.



## **37. A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

cs.CV

21 pages, 7 figures, 10 tables

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2408.10901v2) [paper-pdf](http://arxiv.org/pdf/2408.10901v2)

**Authors**: Zhongliang Guo, Lei Fang, Jingyu Lin, Yifei Qian, Shuai Zhao, Zeyu Wang, Junhao Dong, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in generative AI, particularly Latent Diffusion Models (LDMs), have revolutionized image synthesis and manipulation. However, these generative techniques raises concerns about data misappropriation and intellectual property infringement. Adversarial attacks on machine learning models have been extensively studied, and a well-established body of research has extended these techniques as a benign metric to prevent the underlying misuse of generative AI. Current approaches to safeguarding images from manipulation by LDMs are limited by their reliance on model-specific knowledge and their inability to significantly degrade semantic quality of generated images. In response to these shortcomings, we propose the Posterior Collapse Attack (PCA) based on the observation that VAEs suffer from posterior collapse during training. Our method minimizes dependence on the white-box information of target models to get rid of the implicit reliance on model-specific knowledge. By accessing merely a small amount of LDM parameters, in specific merely the VAE encoder of LDMs, our method causes a substantial semantic collapse in generation quality, particularly in perceptual consistency, and demonstrates strong transferability across various model architectures. Experimental results show that PCA achieves superior perturbation effects on image generation of LDMs with lower runtime and VRAM. Our method outperforms existing techniques, offering a more robust and generalizable solution that is helpful in alleviating the socio-technical challenges posed by the rapidly evolving landscape of generative AI.



## **38. Fisher Information guided Purification against Backdoor Attacks**

cs.CV

Accepted to ACM CCS 2024. arXiv admin note: text overlap with  arXiv:2306.17441

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2409.00863v1) [paper-pdf](http://arxiv.org/pdf/2409.00863v1)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Adnan Siraj Rakin, Zhishan Guo, Nazanin Rahnavard

**Abstract**: Studies on backdoor attacks in recent years suggest that an adversary can compromise the integrity of a deep neural network (DNN) by manipulating a small set of training samples. Our analysis shows that such manipulation can make the backdoor model converge to a bad local minima, i.e., sharper minima as compared to a benign model. Intuitively, the backdoor can be purified by re-optimizing the model to smoother minima. However, a na\"ive adoption of any optimization targeting smoother minima can lead to sub-optimal purification techniques hampering the clean test accuracy. Hence, to effectively obtain such re-optimization, inspired by our novel perspective establishing the connection between backdoor removal and loss smoothness, we propose Fisher Information guided Purification (FIP), a novel backdoor purification framework. Proposed FIP consists of a couple of novel regularizers that aid the model in suppressing the backdoor effects and retaining the acquired knowledge of clean data distribution throughout the backdoor removal procedure through exploiting the knowledge of Fisher Information Matrix (FIM). In addition, we introduce an efficient variant of FIP, dubbed as Fast FIP, which reduces the number of tunable parameters significantly and obtains an impressive runtime gain of almost $5\times$. Extensive experiments show that the proposed method achieves state-of-the-art (SOTA) performance on a wide range of backdoor defense benchmarks: 5 different tasks -- Image Recognition, Object Detection, Video Action Recognition, 3D point Cloud, Language Generation; 11 different datasets including ImageNet, PASCAL VOC, UCF101; diverse model architectures spanning both CNN and vision transformer; 14 different backdoor attacks, e.g., Dynamic, WaNet, LIRA, ISSBA, etc.



## **39. MedFuzz: Exploring the Robustness of Large Language Models in Medical Question Answering**

cs.CL

9 pages, 3 figures, 2 algorithms, appendix

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2406.06573v2) [paper-pdf](http://arxiv.org/pdf/2406.06573v2)

**Authors**: Robert Osazuwa Ness, Katie Matton, Hayden Helm, Sheng Zhang, Junaid Bajwa, Carey E. Priebe, Eric Horvitz

**Abstract**: Large language models (LLM) have achieved impressive performance on medical question-answering benchmarks. However, high benchmark accuracy does not imply that the performance generalizes to real-world clinical settings. Medical question-answering benchmarks rely on assumptions consistent with quantifying LLM performance but that may not hold in the open world of the clinic. Yet LLMs learn broad knowledge that can help the LLM generalize to practical conditions regardless of unrealistic assumptions in celebrated benchmarks. We seek to quantify how well LLM medical question-answering benchmark performance generalizes when benchmark assumptions are violated. Specifically, we present an adversarial method that we call MedFuzz (for medical fuzzing). MedFuzz attempts to modify benchmark questions in ways aimed at confounding the LLM. We demonstrate the approach by targeting strong assumptions about patient characteristics presented in the MedQA benchmark. Successful "attacks" modify a benchmark item in ways that would be unlikely to fool a medical expert but nonetheless "trick" the LLM into changing from a correct to an incorrect answer. Further, we present a permutation test technique that can ensure a successful attack is statistically significant. We show how to use performance on a "MedFuzzed" benchmark, as well as individual successful attacks. The methods show promise at providing insights into the ability of an LLM to operate robustly in more realistic settings.



## **40. GRACE: Graph-Regularized Attentive Convolutional Entanglement with Laplacian Smoothing for Robust DeepFake Video Detection**

cs.CV

Submitted to TPAMI 2024

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2406.19941v3) [paper-pdf](http://arxiv.org/pdf/2406.19941v3)

**Authors**: Chih-Chung Hsu, Shao-Ning Chen, Mei-Hsuan Wu, Yi-Fang Wang, Chia-Ming Lee, Yi-Shiuan Chou

**Abstract**: As DeepFake video manipulation techniques escalate, posing profound threats, the urgent need to develop efficient detection strategies is underscored. However, one particular issue lies with facial images being mis-detected, often originating from degraded videos or adversarial attacks, leading to unexpected temporal artifacts that can undermine the efficacy of DeepFake video detection techniques. This paper introduces a novel method for robust DeepFake video detection, harnessing the power of the proposed Graph-Regularized Attentive Convolutional Entanglement (GRACE) based on the graph convolutional network with graph Laplacian to address the aforementioned challenges. First, conventional Convolution Neural Networks are deployed to perform spatiotemporal features for the entire video. Then, the spatial and temporal features are mutually entangled by constructing a graph with sparse constraint, enforcing essential features of valid face images in the noisy face sequences remaining, thus augmenting stability and performance for DeepFake video detection. Furthermore, the Graph Laplacian prior is proposed in the graph convolutional network to remove the noise pattern in the feature space to further improve the performance. Comprehensive experiments are conducted to illustrate that our proposed method delivers state-of-the-art performance in DeepFake video detection under noisy face sequences. The source code is available at https://github.com/ming053l/GRACE.



## **41. SFR-GNN: Simple and Fast Robust GNNs against Structural Attacks**

cs.LG

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2408.16537v2) [paper-pdf](http://arxiv.org/pdf/2408.16537v2)

**Authors**: Xing Ai, Guanyu Zhu, Yulin Zhu, Yu Zheng, Gaolei Li, Jianhua Li, Kai Zhou

**Abstract**: Graph Neural Networks (GNNs) have demonstrated commendable performance for graph-structured data. Yet, GNNs are often vulnerable to adversarial structural attacks as embedding generation relies on graph topology. Existing efforts are dedicated to purifying the maliciously modified structure or applying adaptive aggregation, thereby enhancing the robustness against adversarial structural attacks. It is inevitable for a defender to consume heavy computational costs due to lacking prior knowledge about modified structures. To this end, we propose an efficient defense method, called Simple and Fast Robust Graph Neural Network (SFR-GNN), supported by mutual information theory. The SFR-GNN first pre-trains a GNN model using node attributes and then fine-tunes it over the modified graph in the manner of contrastive learning, which is free of purifying modified structures and adaptive aggregation, thus achieving great efficiency gains. Consequently, SFR-GNN exhibits a 24%--162% speedup compared to advanced robust models, demonstrating superior robustness for node classification tasks.



## **42. Comprehensive Botnet Detection by Mitigating Adversarial Attacks, Navigating the Subtleties of Perturbation Distances and Fortifying Predictions with Conformal Layers**

cs.CR

46 pages

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2409.00667v1) [paper-pdf](http://arxiv.org/pdf/2409.00667v1)

**Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang

**Abstract**: Botnets are computer networks controlled by malicious actors that present significant cybersecurity challenges. They autonomously infect, propagate, and coordinate to conduct cybercrimes, necessitating robust detection methods. This research addresses the sophisticated adversarial manipulations posed by attackers, aiming to undermine machine learning-based botnet detection systems. We introduce a flow-based detection approach, leveraging machine learning and deep learning algorithms trained on the ISCX and ISOT datasets. The detection algorithms are optimized using the Genetic Algorithm and Particle Swarm Optimization to obtain a baseline detection method. The Carlini & Wagner (C&W) attack and Generative Adversarial Network (GAN) generate deceptive data with subtle perturbations, targeting each feature used for classification while preserving their semantic and syntactic relationships, which ensures that the adversarial samples retain meaningfulness and realism. An in-depth analysis of the required L2 distance from the original sample for the malware sample to misclassify is performed across various iteration checkpoints, showing different levels of misclassification at different L2 distances of the Pertrub sample from the original sample. Our work delves into the vulnerability of various models, examining the transferability of adversarial examples from a Neural Network surrogate model to Tree-based algorithms. Subsequently, models that initially misclassified the perturbed samples are retrained, enhancing their resilience and detection capabilities. In the final phase, a conformal prediction layer is integrated, significantly rejecting incorrect predictions, of 58.20 % in the ISCX dataset and 98.94 % in the ISOT dataset.



## **43. Amplifying Training Data Exposure through Fine-Tuning with Pseudo-Labeled Memberships**

cs.CL

20 pages, 6 figures, 15 tables

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2402.12189v2) [paper-pdf](http://arxiv.org/pdf/2402.12189v2)

**Authors**: Myung Gyo Oh, Hong Eun Ahn, Leo Hyun Park, Taekyoung Kwon

**Abstract**: Neural language models (LMs) are vulnerable to training data extraction attacks due to data memorization. This paper introduces a novel attack scenario wherein an attacker adversarially fine-tunes pre-trained LMs to amplify the exposure of the original training data. This strategy differs from prior studies by aiming to intensify the LM's retention of its pre-training dataset. To achieve this, the attacker needs to collect generated texts that are closely aligned with the pre-training data. However, without knowledge of the actual dataset, quantifying the amount of pre-training data within generated texts is challenging. To address this, we propose the use of pseudo-labels for these generated texts, leveraging membership approximations indicated by machine-generated probabilities from the target LM. We subsequently fine-tune the LM to favor generations with higher likelihoods of originating from the pre-training data, based on their membership probabilities. Our empirical findings indicate a remarkable outcome: LMs with over 1B parameters exhibit a four to eight-fold increase in training data exposure. We discuss potential mitigations and suggest future research directions.



## **44. Robust off-policy Reinforcement Learning via Soft Constrained Adversary**

cs.LG

33 pages, 12 figures, 2 tables

**SubmitDate**: 2024-08-31    [abs](http://arxiv.org/abs/2409.00418v1) [paper-pdf](http://arxiv.org/pdf/2409.00418v1)

**Authors**: Kosuke Nakanishi, Akihiro Kubo, Yuji Yasui, Shin Ishii

**Abstract**: Recently, robust reinforcement learning (RL) methods against input observation have garnered significant attention and undergone rapid evolution due to RL's potential vulnerability. Although these advanced methods have achieved reasonable success, there have been two limitations when considering adversary in terms of long-term horizons. First, the mutual dependency between the policy and its corresponding optimal adversary limits the development of off-policy RL algorithms; although obtaining optimal adversary should depend on the current policy, this has restricted applications to off-policy RL. Second, these methods generally assume perturbations based only on the $L_p$-norm, even when prior knowledge of the perturbation distribution in the environment is available. We here introduce another perspective on adversarial RL: an f-divergence constrained problem with the prior knowledge distribution. From this, we derive two typical attacks and their corresponding robust learning frameworks. The evaluation of robustness is conducted and the results demonstrate that our proposed methods achieve excellent performance in sample-efficient off-policy RL.



## **45. Enhancing Transferability of Adversarial Attacks with GE-AdvGAN+: A Comprehensive Framework for Gradient Editing**

cs.AI

**SubmitDate**: 2024-08-31    [abs](http://arxiv.org/abs/2408.12673v2) [paper-pdf](http://arxiv.org/pdf/2408.12673v2)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Yuchen Zhang, Jiahao Huang, Jianlong Zhou, Fang Chen

**Abstract**: Transferable adversarial attacks pose significant threats to deep neural networks, particularly in black-box scenarios where internal model information is inaccessible. Studying adversarial attack methods helps advance the performance of defense mechanisms and explore model vulnerabilities. These methods can uncover and exploit weaknesses in models, promoting the development of more robust architectures. However, current methods for transferable attacks often come with substantial computational costs, limiting their deployment and application, especially in edge computing scenarios. Adversarial generative models, such as Generative Adversarial Networks (GANs), are characterized by their ability to generate samples without the need for retraining after an initial training phase. GE-AdvGAN, a recent method for transferable adversarial attacks, is based on this principle. In this paper, we propose a novel general framework for gradient editing-based transferable attacks, named GE-AdvGAN+, which integrates nearly all mainstream attack methods to enhance transferability while significantly reducing computational resource consumption. Our experiments demonstrate the compatibility and effectiveness of our framework. Compared to the baseline AdvGAN, our best-performing method, GE-AdvGAN++, achieves an average ASR improvement of 47.8. Additionally, it surpasses the latest competing algorithm, GE-AdvGAN, with an average ASR increase of 5.9. The framework also exhibits enhanced computational efficiency, achieving 2217.7 FPS, outperforming traditional methods such as BIM and MI-FGSM. The implementation code for our GE-AdvGAN+ framework is available at https://github.com/GEAdvGANP



## **46. LightPure: Realtime Adversarial Image Purification for Mobile Devices Using Diffusion Models**

cs.CR

**SubmitDate**: 2024-08-31    [abs](http://arxiv.org/abs/2409.00340v1) [paper-pdf](http://arxiv.org/pdf/2409.00340v1)

**Authors**: Hossein Khalili, Seongbin Park, Vincent Li, Brandan Bright, Ali Payani, Ramana Rao Kompella, Nader Sehatbakhsh

**Abstract**: Autonomous mobile systems increasingly rely on deep neural networks for perception and decision-making. While effective, these systems are vulnerable to adversarial machine learning attacks where minor input perturbations can significantly impact outcomes. Common countermeasures involve adversarial training and/or data or network transformation. These methods, though effective, require full access to typically proprietary classifiers and are costly for large models. Recent solutions propose purification models, which add a "purification" layer before classification, eliminating the need to modify the classifier directly. Despite their effectiveness, these methods are compute-intensive, making them unsuitable for mobile systems where resources are limited and low latency is essential.   This paper introduces LightPure, a new method that enhances adversarial image purification. It improves the accuracy of existing purification methods and provides notable enhancements in speed and computational efficiency, making it suitable for mobile devices with limited resources. Our approach uses a two-step diffusion and one-shot Generative Adversarial Network (GAN) framework, prioritizing latency without compromising robustness. We propose several new techniques to achieve a reasonable balance between classification accuracy and adversarial robustness while maintaining desired latency. We design and implement a proof-of-concept on a Jetson Nano board and evaluate our method using various attack scenarios and datasets. Our results show that LightPure can outperform existing methods by up to 10x in terms of latency while achieving higher accuracy and robustness for various attack scenarios. This method offers a scalable and effective solution for real-world mobile systems.



## **47. Exact Recovery Guarantees for Parameterized Non-linear System Identification Problem under Adversarial Attacks**

math.OC

33 pages

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2409.00276v1) [paper-pdf](http://arxiv.org/pdf/2409.00276v1)

**Authors**: Haixiang Zhang, Baturalp Yalcin, Javad Lavaei, Eduardo Sontag

**Abstract**: In this work, we study the system identification problem for parameterized non-linear systems using basis functions under adversarial attacks. Motivated by the LASSO-type estimators, we analyze the exact recovery property of a non-smooth estimator, which is generated by solving an embedded $\ell_1$-loss minimization problem. First, we derive necessary and sufficient conditions for the well-specifiedness of the estimator and the uniqueness of global solutions to the underlying optimization problem. Next, we provide exact recovery guarantees for the estimator under two different scenarios of boundedness and Lipschitz continuity of the basis functions. The non-asymptotic exact recovery is guaranteed with high probability, even when there are more severely corrupted data than clean data. Finally, we numerically illustrate the validity of our theory. This is the first study on the sample complexity analysis of a non-smooth estimator for the non-linear system identification problem.



## **48. RandOhm: Mitigating Impedance Side-channel Attacks using Randomized Circuit Configurations**

cs.CR

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2401.08925v3) [paper-pdf](http://arxiv.org/pdf/2401.08925v3)

**Authors**: Saleh Khalaj Monfared, Domenic Forte, Shahin Tajik

**Abstract**: Physical side-channel attacks can compromise the security of integrated circuits. Most physical side-channel attacks (e.g., power or electromagnetic) exploit the dynamic behavior of a chip, typically manifesting as changes in current consumption or voltage fluctuations where algorithmic countermeasures, such as masking, can effectively mitigate them. However, as demonstrated recently, these mitigation techniques are not entirely effective against backscattered side-channel attacks such as impedance analysis. In the case of an impedance attack, an adversary exploits the data-dependent impedance variations of the chip power delivery network (PDN) to extract secret information. In this work, we introduce RandOhm, which exploits a moving target defense (MTD) strategy based on the partial reconfiguration (PR) feature of mainstream FPGAs and programmable SoCs to defend against impedance side-channel attacks. We demonstrate that the information leakage through the PDN impedance could be significantly reduced via runtime reconfiguration of the secret-sensitive parts of the circuitry. Hence, by constantly randomizing the placement and routing of the circuit, one can decorrelate the data-dependent computation from the impedance value. Moreover, in contrast to existing PR-based countermeasures, RandOhm deploys open-source bitstream manipulation tools on programmable SoCs to speed up the randomization and provide real-time protection. To validate our claims, we apply RandOhm to AES ciphers realized on 28-nm FPGAs. We analyze the resiliency of our approach by performing non-profiled and profiled impedance analysis attacks and investigate the overhead of our mitigation in terms of delay and performance.



## **49. ARINC 429 Cyber-vulnerabilities and Voltage Data in a Hardware-in-the-Loop Simulator**

cs.CR

7 pages, 3 figures. Intended for publication in IEEE Transactions on  Aerospace and Electronic Systems

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2408.16714v2) [paper-pdf](http://arxiv.org/pdf/2408.16714v2)

**Authors**: Connor Trask, Steve Movit, Justace Clutter, Rosene Clark, Mark Herrera, Kelly Tran

**Abstract**: ARINC 429 is a ubiquitous data bus for civil avionics, enabling reliable communication between devices from disparate manufacturers. However, ARINC 429 lacks any form of encryption or authentication, making it an inherently insecure communication protocol and rendering any connected avionics vulnerable to a range of attacks. We constructed a hardware-in-the-loop simulator with ARINC 429 buses, explored these vulnerabilities, and identified their potential to deny, degrade, or disrupt aircraft capabilities. We performed a denial-of-service attack against a multi-function display via a compromised ARINC 429 bus using commercially available tools, which succeeded in disabling important navigational aids. This proven attack on physical avionics illustrates the risk inherent in ARINC 429 and the need for the ability to detect these attacks. One potential mitigation is an intrusion detection system (IDS) trained on data collected from the electrical properties of the physical bus. Although previous research has demonstrated the feasibility of an IDS on an ARINC 429 bus, no IDS has been trained on data generated by avionics hardware. To facilitate this, we recorded voltage traces and message history generated by avionics and adversarial devices on the ARINC 429 bus. To the best of our knowledge, this is the first publicly available collection of hardware-generated ARINC 429 signal data.



## **50. Jailbreak Attacks and Defenses Against Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2407.04295v2) [paper-pdf](http://arxiv.org/pdf/2407.04295v2)

**Authors**: Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, Qi Li

**Abstract**: Large Language Models (LLMs) have performed exceptionally in various text-generative tasks, including question answering, translation, code completion, etc. However, the over-assistance of LLMs has raised the challenge of "jailbreaking", which induces the model to generate malicious responses against the usage policy and society by designing adversarial prompts. With the emergence of jailbreak attack methods exploiting different vulnerabilities in LLMs, the corresponding safety alignment measures are also evolving. In this paper, we propose a comprehensive and detailed taxonomy of jailbreak attack and defense methods. For instance, the attack methods are divided into black-box and white-box attacks based on the transparency of the target model. Meanwhile, we classify defense methods into prompt-level and model-level defenses. Additionally, we further subdivide these attack and defense methods into distinct sub-classes and present a coherent diagram illustrating their relationships. We also conduct an investigation into the current evaluation methods and compare them from different perspectives. Our findings aim to inspire future research and practical implementations in safeguarding LLMs against adversarial attacks. Above all, although jailbreak remains a significant concern within the community, we believe that our work enhances the understanding of this domain and provides a foundation for developing more secure LLMs.



