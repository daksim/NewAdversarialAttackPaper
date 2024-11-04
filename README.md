# Latest Adversarial Attack Papers
**update at 2024-11-04 11:09:37**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Efficient Adversarial Training in LLMs with Continuous Attacks**

cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15589v3) [paper-pdf](http://arxiv.org/pdf/2405.15589v3)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on five models from different families (Gemma, Phi3, Mistral, Zephyr, Llama2) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.



## **2. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

math.OC

Theorem 1 turned out to be incorrect

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2410.03218v2) [paper-pdf](http://arxiv.org/pdf/2410.03218v2)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are completely arbitrary. We show that when the probability of having an attack at a given time is less than 0.5, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.



## **3. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16405v2) [paper-pdf](http://arxiv.org/pdf/2405.16405v2)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.



## **4. Improved Generation of Adversarial Examples Against Safety-aligned LLMs**

cs.CR

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.20778v2) [paper-pdf](http://arxiv.org/pdf/2405.20778v2)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Adversarial prompts generated using gradient-based methods exhibit outstanding performance in performing automatic jailbreak attacks against safety-aligned LLMs. Nevertheless, due to the discrete nature of texts, the input gradient of LLMs struggles to precisely reflect the magnitude of loss change that results from token replacements in the prompt, leading to limited attack success rates against safety-aligned LLMs, even in the white-box setting. In this paper, we explore a new perspective on this problem, suggesting that it can be alleviated by leveraging innovations inspired in transfer-based attacks that were originally proposed for attacking black-box image classification models. For the first time, we appropriate the ideologies of effective methods among these transfer-based attacks, i.e., Skip Gradient Method and Intermediate Level Attack, into gradient-based adversarial prompt generation and achieve significant performance gains without introducing obvious computational cost. Meanwhile, by discussing mechanisms behind the gains, new insights are drawn, and proper combinations of these methods are also developed. Our empirical results show that 87% of the query-specific adversarial suffixes generated by the developed combination can induce Llama-2-7B-Chat to produce the output that exactly matches the target string on AdvBench. This match rate is 33% higher than that of a very strong baseline known as GCG, demonstrating advanced discrete optimization for adversarial prompt generation against LLMs. In addition, without introducing obvious cost, the combination achieves >30% absolute increase in attack success rates compared with GCG when generating both query-specific (38% -> 68%) and universal adversarial prompts (26.68% -> 60.32%) for attacking the Llama-2-7B-Chat model on AdvBench. Code at: https://github.com/qizhangli/Gradient-based-Jailbreak-Attacks.



## **5. Uncertainty-based Offline Variational Bayesian Reinforcement Learning for Robustness under Diverse Data Corruptions**

cs.LG

Accepted to NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00465v1) [paper-pdf](http://arxiv.org/pdf/2411.00465v1)

**Authors**: Rui Yang, Jie Wang, Guoping Wu, Bin Li

**Abstract**: Real-world offline datasets are often subject to data corruptions (such as noise or adversarial attacks) due to sensor failures or malicious attacks. Despite advances in robust offline reinforcement learning (RL), existing methods struggle to learn robust agents under high uncertainty caused by the diverse corrupted data (i.e., corrupted states, actions, rewards, and dynamics), leading to performance degradation in clean environments. To tackle this problem, we propose a novel robust variational Bayesian inference for offline RL (TRACER). It introduces Bayesian inference for the first time to capture the uncertainty via offline data for robustness against all types of data corruptions. Specifically, TRACER first models all corruptions as the uncertainty in the action-value function. Then, to capture such uncertainty, it uses all offline data as the observations to approximate the posterior distribution of the action-value function under a Bayesian inference framework. An appealing feature of TRACER is that it can distinguish corrupted data from clean data using an entropy-based uncertainty measure, since corrupted data often induces higher uncertainty and entropy. Based on the aforementioned measure, TRACER can regulate the loss associated with corrupted data to reduce its influence, thereby enhancing robustness and performance in clean environments. Experiments demonstrate that TRACER significantly outperforms several state-of-the-art approaches across both individual and simultaneous data corruptions.



## **6. Adversarial Purification and Fine-tuning for Robust UDC Image Restoration**

eess.IV

Failure to meet expectations

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2402.13629v3) [paper-pdf](http://arxiv.org/pdf/2402.13629v3)

**Authors**: Zhenbo Song, Zhenyuan Zhang, Kaihao Zhang, Zhaoxin Fan, Jianfeng Lu

**Abstract**: This study delves into the enhancement of Under-Display Camera (UDC) image restoration models, focusing on their robustness against adversarial attacks. Despite its innovative approach to seamless display integration, UDC technology faces unique image degradation challenges exacerbated by the susceptibility to adversarial perturbations. Our research initially conducts an in-depth robustness evaluation of deep-learning-based UDC image restoration models by employing several white-box and black-box attacking methods. This evaluation is pivotal in understanding the vulnerabilities of current UDC image restoration techniques. Following the assessment, we introduce a defense framework integrating adversarial purification with subsequent fine-tuning processes. First, our approach employs diffusion-based adversarial purification, effectively neutralizing adversarial perturbations. Then, we apply the fine-tuning methodologies to refine the image restoration models further, ensuring that the quality and fidelity of the restored images are maintained. The effectiveness of our proposed approach is validated through extensive experiments, showing marked improvements in resilience against typical adversarial attacks.



## **7. Towards Building Secure UAV Navigation with FHE-aware Knowledge Distillation**

cs.CR

arXiv admin note: text overlap with arXiv:2404.17225

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00403v1) [paper-pdf](http://arxiv.org/pdf/2411.00403v1)

**Authors**: Arjun Ramesh Kaushik, Charanjit Jutla, Nalini Ratha

**Abstract**: In safeguarding mission-critical systems, such as Unmanned Aerial Vehicles (UAVs), preserving the privacy of path trajectories during navigation is paramount. While the combination of Reinforcement Learning (RL) and Fully Homomorphic Encryption (FHE) holds promise, the computational overhead of FHE presents a significant challenge. This paper proposes an innovative approach that leverages Knowledge Distillation to enhance the practicality of secure UAV navigation. By integrating RL and FHE, our framework addresses vulnerabilities to adversarial attacks while enabling real-time processing of encrypted UAV camera feeds, ensuring data security. To mitigate FHE's latency, Knowledge Distillation is employed to compress the network, resulting in an impressive 18x speedup without compromising performance, as evidenced by an R-squared score of 0.9499 compared to the original model's score of 0.9631. Our methodology underscores the feasibility of processing encrypted data for UAV navigation tasks, emphasizing security alongside performance efficiency and timely processing. These findings pave the way for deploying autonomous UAVs in sensitive environments, bolstering their resilience against potential security threats.



## **8. OSLO: One-Shot Label-Only Membership Inference Attacks**

cs.LG

To appear at NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16978v3) [paper-pdf](http://arxiv.org/pdf/2405.16978v3)

**Authors**: Yuefeng Peng, Jaechul Roh, Subhransu Maji, Amir Houmansadr

**Abstract**: We introduce One-Shot Label-Only (OSLO) membership inference attacks (MIAs), which accurately infer a given sample's membership in a target model's training set with high precision using just \emph{a single query}, where the target model only returns the predicted hard label. This is in contrast to state-of-the-art label-only attacks which require $\sim6000$ queries, yet get attack precisions lower than OSLO's. OSLO leverages transfer-based black-box adversarial attacks. The core idea is that a member sample exhibits more resistance to adversarial perturbations than a non-member. We compare OSLO against state-of-the-art label-only attacks and demonstrate that, despite requiring only one query, our method significantly outperforms previous attacks in terms of precision and true positive rate (TPR) under the same false positive rates (FPR). For example, compared to previous label-only MIAs, OSLO achieves a TPR that is at least 7$\times$ higher under a 1\% FPR and at least 22$\times$ higher under a 0.1\% FPR on CIFAR100 for a ResNet18 model. We evaluated multiple defense mechanisms against OSLO.



## **9. Quantum Entanglement Path Selection and Qubit Allocation via Adversarial Group Neural Bandits**

quant-ph

Accepted by IEEE/ACM Transactions on Networking

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00316v1) [paper-pdf](http://arxiv.org/pdf/2411.00316v1)

**Authors**: Yin Huang, Lei Wang, Jie Xu

**Abstract**: Quantum Data Networks (QDNs) have emerged as a promising framework in the field of information processing and transmission, harnessing the principles of quantum mechanics. QDNs utilize a quantum teleportation technique through long-distance entanglement connections, encoding data information in quantum bits (qubits). Despite being a cornerstone in various quantum applications, quantum entanglement encounters challenges in establishing connections over extended distances due to probabilistic processes influenced by factors like optical fiber losses. The creation of long-distance entanglement connections between quantum computers involves multiple entanglement links and entanglement swapping techniques through successive quantum nodes, including quantum computers and quantum repeaters, necessitating optimal path selection and qubit allocation. Current research predominantly assumes known success rates of entanglement links between neighboring quantum nodes and overlooks potential network attackers. This paper addresses the online challenge of optimal path selection and qubit allocation, aiming to learn the best strategy for achieving the highest success rate of entanglement connections between two chosen quantum computers without prior knowledge of the success rate and in the presence of a QDN attacker. The proposed approach is based on multi-armed bandits, specifically adversarial group neural bandits, which treat each path as a group and view qubit allocation as arm selection. Our contributions encompass formulating an online adversarial optimization problem, introducing the EXPNeuralUCB bandits algorithm with theoretical performance guarantees, and conducting comprehensive simulations to showcase its superiority over established advanced algorithms.



## **10. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

cs.LG

10 pages, 6 figures, 2 tables. Version Update: Neurips Camera Ready

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2406.18451v3) [paper-pdf](http://arxiv.org/pdf/2406.18451v3)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate high margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively and confidently use the logit margin to detect brittle decisions with such models. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to assess adversarial vulnerability in deployment scenarios efficiently.



## **11. Efficient Model Compression for Bayesian Neural Networks**

cs.LG

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00273v1) [paper-pdf](http://arxiv.org/pdf/2411.00273v1)

**Authors**: Diptarka Saha, Zihe Liu, Feng Liang

**Abstract**: Model Compression has drawn much attention within the deep learning community recently. Compressing a dense neural network offers many advantages including lower computation cost, deployability to devices of limited storage and memories, and resistance to adversarial attacks. This may be achieved via weight pruning or fully discarding certain input features. Here we demonstrate a novel strategy to emulate principles of Bayesian model selection in a deep learning setup. Given a fully connected Bayesian neural network with spike-and-slab priors trained via a variational algorithm, we obtain the posterior inclusion probability for every node that typically gets lost. We employ these probabilities for pruning and feature selection on a host of simulated and real-world benchmark data and find evidence of better generalizability of the pruned model in all our experiments.



## **12. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

The camera-ready version of JailbreakBench v1.0 (accepted at NeurIPS  2024 Datasets and Benchmarks Track): more attack artifacts, more test-time  defenses, a more accurate jailbreak judge (Llama-3-70B with a custom prompt),  a larger dataset of human preferences for selecting a jailbreak judge (300  examples), an over-refusal evaluation dataset, a semantic refusal judge based  on Llama-3-8B

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2404.01318v5) [paper-pdf](http://arxiv.org/pdf/2404.01318v5)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.



## **13. Protecting Feed-Forward Networks from Adversarial Attacks Using Predictive Coding**

cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00222v1) [paper-pdf](http://arxiv.org/pdf/2411.00222v1)

**Authors**: Ehsan Ganjidoost, Jeff Orchard

**Abstract**: An adversarial example is a modified input image designed to cause a Machine Learning (ML) model to make a mistake; these perturbations are often invisible or subtle to human observers and highlight vulnerabilities in a model's ability to generalize from its training data. Several adversarial attacks can create such examples, each with a different perspective, effectiveness, and perceptibility of changes. Conversely, defending against such adversarial attacks improves the robustness of ML models in image processing and other domains of deep learning. Most defence mechanisms require either a level of model awareness, changes to the model, or access to a comprehensive set of adversarial examples during training, which is impractical. Another option is to use an auxiliary model in a preprocessing manner without changing the primary model. This study presents a practical and effective solution -- using predictive coding networks (PCnets) as an auxiliary step for adversarial defence. By seamlessly integrating PCnets into feed-forward networks as a preprocessing step, we substantially bolster resilience to adversarial perturbations. Our experiments on MNIST and CIFAR10 demonstrate the remarkable effectiveness of PCnets in mitigating adversarial examples with about 82% and 65% improvements in robustness, respectively. The PCnet, trained on a small subset of the dataset, leverages its generative nature to effectively counter adversarial efforts, reverting perturbed images closer to their original forms. This innovative approach holds promise for enhancing the security and reliability of neural network classifiers in the face of the escalating threat of adversarial attacks.



## **14. I Can Hear You: Selective Robust Training for Deepfake Audio Detection**

cs.SD

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00121v1) [paper-pdf](http://arxiv.org/pdf/2411.00121v1)

**Authors**: Zirui Zhang, Wei Hao, Aroon Sankoh, William Lin, Emanuel Mendiola-Ortiz, Junfeng Yang, Chengzhi Mao

**Abstract**: Recent advances in AI-generated voices have intensified the challenge of detecting deepfake audio, posing risks for scams and the spread of disinformation. To tackle this issue, we establish the largest public voice dataset to date, named DeepFakeVox-HQ, comprising 1.3 million samples, including 270,000 high-quality deepfake samples from 14 diverse sources. Despite previously reported high accuracy, existing deepfake voice detectors struggle with our diversely collected dataset, and their detection success rates drop even further under realistic corruptions and adversarial attacks. We conduct a holistic investigation into factors that enhance model robustness and show that incorporating a diversified set of voice augmentations is beneficial. Moreover, we find that the best detection models often rely on high-frequency features, which are imperceptible to humans and can be easily manipulated by an attacker. To address this, we propose the F-SAT: Frequency-Selective Adversarial Training method focusing on high-frequency components. Empirical results demonstrate that using our training dataset boosts baseline model performance (without robust training) by 33%, and our robust training further improves accuracy by 7.7% on clean samples and by 29.3% on corrupted and attacked samples, over the state-of-the-art RawNet3 model.



## **15. Untelegraphable Encryption and its Applications**

quant-ph

55 pages

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.24189v1) [paper-pdf](http://arxiv.org/pdf/2410.24189v1)

**Authors**: Jeffrey Champion, Fuyuki Kitagawa, Ryo Nishimaki, Takashi Yamakawa

**Abstract**: We initiate the study of untelegraphable encryption (UTE), founded on the no-telegraphing principle, which allows an encryptor to encrypt a message such that a binary string representation of the ciphertext cannot be decrypted by a user with the secret key, a task that is classically impossible. This is a natural relaxation of unclonable encryption, inspired by the recent work of Nehoran and Zhandry (ITCS 2024), who showed a computational separation between the no-cloning and no-telegraphing principles. In this work, we define and construct UTE information-theoretically in the plain model. Building off this, we give several applications of UTE and study the interplay of UTE with UE and well-studied tasks in quantum state learning, yielding the following contributions:   - A construction of collusion-resistant UTE from standard secret-key encryption. We additionally show that hyper-efficient shadow tomography (HEST) is impossible assuming collusion-resistant UTE exists. By considering a relaxation of collusion-resistant UTE, we are able to show the impossibility of HEST assuming only pseudorandom state generators (which may not imply one-way functions). This almost completely answers an open inquiry of Aaronson (STOC 2019).   - A construction of UTE from a one-shot message authentication code in the classical oracle model, such that there is an explicit attack that breaks UE security for an unbounded polynomial number of decryptors.   - A construction of everlasting secure collusion-resistant UTE, where the decryptor adversary can run in unbounded time, in the quantum random oracle model (QROM), and formal evidence that a construction in the plain model is a challenging task. We additionally show that HEST with unbounded post-processing time is impossible in the QROM.   - Constructions (and definitions) of untelegraphable secret sharing and untelegraphable functional encryption.



## **16. Unveiling Synthetic Faces: How Synthetic Datasets Can Expose Real Identities**

cs.CV

Accepted in NeurIPS 2024 Workshop on New Frontiers in Adversarial  Machine Learning

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.24015v1) [paper-pdf](http://arxiv.org/pdf/2410.24015v1)

**Authors**: Hatef Otroshi Shahreza, Sébastien Marcel

**Abstract**: Synthetic data generation is gaining increasing popularity in different computer vision applications. Existing state-of-the-art face recognition models are trained using large-scale face datasets, which are crawled from the Internet and raise privacy and ethical concerns. To address such concerns, several works have proposed generating synthetic face datasets to train face recognition models. However, these methods depend on generative models, which are trained on real face images. In this work, we design a simple yet effective membership inference attack to systematically study if any of the existing synthetic face recognition datasets leak any information from the real data used to train the generator model. We provide an extensive study on 6 state-of-the-art synthetic face recognition datasets, and show that in all these synthetic datasets, several samples from the original real dataset are leaked. To our knowledge, this paper is the first work which shows the leakage from training data of generator models into the generated synthetic face recognition datasets. Our study demonstrates privacy pitfalls in synthetic face recognition datasets and paves the way for future studies on generating responsible synthetic face datasets.



## **17. DiffPAD: Denoising Diffusion-based Adversarial Patch Decontamination**

cs.CV

Accepted to 2025 IEEE/CVF Winter Conference on Applications of  Computer Vision (WACV)

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.24006v1) [paper-pdf](http://arxiv.org/pdf/2410.24006v1)

**Authors**: Jia Fu, Xiao Zhang, Sepideh Pashami, Fatemeh Rahimian, Anders Holst

**Abstract**: In the ever-evolving adversarial machine learning landscape, developing effective defenses against patch attacks has become a critical challenge, necessitating reliable solutions to safeguard real-world AI systems. Although diffusion models have shown remarkable capacity in image synthesis and have been recently utilized to counter $\ell_p$-norm bounded attacks, their potential in mitigating localized patch attacks remains largely underexplored. In this work, we propose DiffPAD, a novel framework that harnesses the power of diffusion models for adversarial patch decontamination. DiffPAD first performs super-resolution restoration on downsampled input images, then adopts binarization, dynamic thresholding scheme and sliding window for effective localization of adversarial patches. Such a design is inspired by the theoretically derived correlation between patch size and diffusion restoration error that is generalized across diverse patch attack scenarios. Finally, DiffPAD applies inpainting techniques to the original input images with the estimated patch region being masked. By integrating closed-form solutions for super-resolution restoration and image inpainting into the conditional reverse sampling process of a pre-trained diffusion model, DiffPAD obviates the need for text guidance or fine-tuning. Through comprehensive experiments, we demonstrate that DiffPAD not only achieves state-of-the-art adversarial robustness against patch attacks but also excels in recovering naturalistic images without patch remnants.



## **18. Meta-Learning Approaches for Improving Detection of Unseen Speech Deepfakes**

eess.AS

6 pages, accepted to the IEEE Spoken Language Technology Workshop  (SLT) 2024

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.20578v2) [paper-pdf](http://arxiv.org/pdf/2410.20578v2)

**Authors**: Ivan Kukanov, Janne Laakkonen, Tomi Kinnunen, Ville Hautamäki

**Abstract**: Current speech deepfake detection approaches perform satisfactorily against known adversaries; however, generalization to unseen attacks remains an open challenge. The proliferation of speech deepfakes on social media underscores the need for systems that can generalize to unseen attacks not observed during training. We address this problem from the perspective of meta-learning, aiming to learn attack-invariant features to adapt to unseen attacks with very few samples available. This approach is promising since generating of a high-scale training dataset is often expensive or infeasible. Our experiments demonstrated an improvement in the Equal Error Rate (EER) from 21.67% to 10.42% on the InTheWild dataset, using just 96 samples from the unseen dataset. Continuous few-shot adaptation ensures that the system remains up-to-date.



## **19. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

cs.LG

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2402.06255v4) [paper-pdf](http://arxiv.org/pdf/2402.06255v4)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreaking attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly focusing on model fine-tuning or heuristical defense designs. However, how to achieve intrinsic robustness through prompt optimization remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0%, while maintaining the model's utility on the benign task and incurring only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/PKU-ML/PAT.



## **20. Noise as a Double-Edged Sword: Reinforcement Learning Exploits Randomized Defenses in Neural Networks**

cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23870v1) [paper-pdf](http://arxiv.org/pdf/2410.23870v1)

**Authors**: Steve Bakos, Pooria Madani, Heidar Davoudi

**Abstract**: This study investigates a counterintuitive phenomenon in adversarial machine learning: the potential for noise-based defenses to inadvertently aid evasion attacks in certain scenarios. While randomness is often employed as a defensive strategy against adversarial examples, our research reveals that this approach can sometimes backfire, particularly when facing adaptive attackers using reinforcement learning (RL). Our findings show that in specific cases, especially with visually noisy classes, the introduction of noise in the classifier's confidence values can be exploited by the RL attacker, leading to a significant increase in evasion success rates. In some instances, the noise-based defense scenario outperformed other strategies by up to 20\% on a subset of classes. However, this effect was not consistent across all classifiers tested, highlighting the complexity of the interaction between noise-based defenses and different models. These results suggest that in some cases, noise-based defenses can inadvertently create an adversarial training loop beneficial to the RL attacker. Our study emphasizes the need for a more nuanced approach to defensive strategies in adversarial machine learning, particularly in safety-critical applications. It challenges the assumption that randomness universally enhances defense against evasion attacks and highlights the importance of considering adaptive, RL-based attackers when designing robust defense mechanisms.



## **21. Rapid Plug-in Defenders**

cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2306.01762v4) [paper-pdf](http://arxiv.org/pdf/2306.01762v4)

**Authors**: Kai Wu, Yujian Betterest Li, Jian Lou, Xiaoyu Zhang, Handing Wang, Jing Liu

**Abstract**: In the realm of daily services, the deployment of deep neural networks underscores the paramount importance of their reliability. However, the vulnerability of these networks to adversarial attacks, primarily evasion-based, poses a concerning threat to their functionality. Common methods for enhancing robustness involve heavy adversarial training or leveraging learned knowledge from clean data, both necessitating substantial computational resources. This inherent time-intensive nature severely limits the agility of large foundational models to swiftly counter adversarial perturbations. To address this challenge, this paper focuses on the Rapid Plug-in Defender (RaPiD) problem, aiming to rapidly counter adversarial perturbations without altering the deployed model. Drawing inspiration from the generalization and the universal computation ability of pre-trained transformer models, we propose a novel method termed CeTaD (Considering Pre-trained Transformers as Defenders) for RaPiD, optimized for efficient computation. CeTaD strategically fine-tunes the normalization layer parameters within the defender using a limited set of clean and adversarial examples. Our evaluation centers on assessing CeTaD's effectiveness, transferability, and the impact of different components in scenarios involving one-shot adversarial examples. The proposed method is capable of rapidly adapting to various attacks and different application scenarios without altering the target model and clean training data. We also explore the influence of varying training data conditions on CeTaD's performance. Notably, CeTaD exhibits adaptability across differentiable service models and proves the potential of continuous learning.



## **22. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

cs.CL

Accepted to NeurIPS 2024 Dataset & Benchmarking Track

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23746v1) [paper-pdf](http://arxiv.org/pdf/2410.23746v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating advanced prompt usages, human revisions like word substitutions, and writing errors. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.



## **23. One Prompt to Verify Your Models: Black-Box Text-to-Image Models Verification via Non-Transferable Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.22725v2) [paper-pdf](http://arxiv.org/pdf/2410.22725v2)

**Authors**: Ji Guo, Wenbo Jiang, Rui Zhang, Guoming Lu, Hongwei Li

**Abstract**: Recently, the success of Text-to-Image (T2I) models has led to the rise of numerous third-party platforms, which claim to provide cheaper API services and more flexibility in model options. However, this also raises a new security concern: Are these third-party services truly offering the models they claim? To address this problem, we propose the first T2I model verification method named Text-to-Image Model Verification via Non-Transferable Adversarial Attacks (TVN). The non-transferability of adversarial examples means that these examples are only effective on a target model and ineffective on other models, thereby allowing for the verification of the target model. TVN utilizes the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize the cosine similarity of a prompt's text encoding, generating non-transferable adversarial prompts. By calculating the CLIP-text scores between the non-transferable adversarial prompts without perturbations and the images, we can verify if the model matches the claimed target model, based on a 3-sigma threshold. The experiments showed that TVN performed well in both closed-set and open-set scenarios, achieving a verification accuracy of over 90\%. Moreover, the adversarial prompts generated by TVN significantly reduced the CLIP-text scores of the target model, while having little effect on other models.



## **24. The Influence of Ridership Weighting on Targeting and Recovery Strategies for Urban Rail Rapid Transit Systems**

physics.soc-ph

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23688v1) [paper-pdf](http://arxiv.org/pdf/2410.23688v1)

**Authors**: Aran Chakraborty, Yushi Tsukimoto, August Posch, Jack Watson, Auroop Ganguly

**Abstract**: The resilience of urban rapid transit systems (URTs) to a rapidly evolving threat space is of much concern. Extreme rainfall events are both intensifying and growing more frequent under continuing climate change, exposing transit systems to flooding, while cyber threats and emerging technologies such as unmanned aerial vehicles are exposing such systems to targeted disruptions. An imperative has emerged to model how networked infrastructure systems fail and devise strategies to efficiently recover from disruptions. Passenger flow approaches can quantify more dimensions of resilience than network science-based approaches, but the former typically requires granular data from automatic fare collection and suffers from large runtime complexities. Some attempts have been made to include accessible low-resolution ridership data in topological frameworks. However, there is yet to be a systematic investigation of the effects of incorporating low-dimensional, coarsely-averaged ridership volume into topological network science methodologies. We simulate targeted attack and recovery sequences using station-level ridership, four centrality measures, and weighted combinations thereof. Resilience is quantified using two topological measures of performance: the node count of a network's giant connected component (GCC), and a new measure termed the "highest ridership connected component" (HRCC). Three transit systems are used as case studies: the subways of Boston, New York, and Osaka. Results show that centrality-based strategies are most effective when measuring performance via GCC, while centrality-ridership hybrid strategies perform strongest by HRCC. We show that the most effective strategies vary by network characteristics and the mission goals of emergency managers, highlighting the need to plan for strategic adversaries and rapid recovery according to each city's unique needs.



## **25. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

cs.CV

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23687v1) [paper-pdf](http://arxiv.org/pdf/2410.23687v1)

**Authors**: Chiyu Zhang, Xiaogang Xu, Jiafei Wu, Zhe Liu, Lu Zhou

**Abstract**: Adversarial attacks, which manipulate input data to undermine model availability and integrity, pose significant security threats during machine learning inference. With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreak techniques, have emerged. Understanding these attacks is crucial for developing more robust systems and demystifying the inner workings of neural networks. However, existing reviews often focus on attack classifications and lack comprehensive, in-depth analysis. The research community currently needs: 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations of existing methods; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.



## **26. Pseudo-Conversation Injection for LLM Goal Hijacking**

cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23678v1) [paper-pdf](http://arxiv.org/pdf/2410.23678v1)

**Authors**: Zheng Chen, Buhui Yao

**Abstract**: Goal hijacking is a type of adversarial attack on Large Language Models (LLMs) where the objective is to manipulate the model into producing a specific, predetermined output, regardless of the user's original input. In goal hijacking, an attacker typically appends a carefully crafted malicious suffix to the user's prompt, which coerces the model into ignoring the user's original input and generating the target response. In this paper, we introduce a novel goal hijacking attack method called Pseudo-Conversation Injection, which leverages the weaknesses of LLMs in role identification within conversation contexts. Specifically, we construct the suffix by fabricating responses from the LLM to the user's initial prompt, followed by a prompt for a malicious new task. This leads the model to perceive the initial prompt and fabricated response as a completed conversation, thereby executing the new, falsified prompt. Following this approach, we propose three Pseudo-Conversation construction strategies: Targeted Pseudo-Conversation, Universal Pseudo-Conversation, and Robust Pseudo-Conversation. These strategies are designed to achieve effective goal hijacking across various scenarios. Our experiments, conducted on two mainstream LLM platforms including ChatGPT and Qwen, demonstrate that our proposed method significantly outperforms existing approaches in terms of attack effectiveness.



## **27. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

cs.SE

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2308.11161v2) [paper-pdf](http://arxiv.org/pdf/2308.11161v2)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.



## **28. Keep on Swimming: Real Attackers Only Need Partial Knowledge of a Multi-Model System**

cs.LG

11 pages, 2 figures

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23483v1) [paper-pdf](http://arxiv.org/pdf/2410.23483v1)

**Authors**: Julian Collado, Kevin Stangl

**Abstract**: Recent approaches in machine learning often solve a task using a composition of multiple models or agentic architectures. When targeting a composed system with adversarial attacks, it might not be computationally or informationally feasible to train an end-to-end proxy model or a proxy model for every component of the system. We introduce a method to craft an adversarial attack against the overall multi-model system when we only have a proxy model for the final black-box model, and when the transformation applied by the initial models can make the adversarial perturbations ineffective. Current methods handle this by applying many copies of the first model/transformation to an input and then re-use a standard adversarial attack by averaging gradients, or learning a proxy model for both stages. To our knowledge, this is the first attack specifically designed for this threat model and our method has a substantially higher attack success rate (80% vs 25%) and contains 9.4% smaller perturbations (MSE) compared to prior state-of-the-art methods. Our experiments focus on a supervised image pipeline, but we are confident the attack will generalize to other multi-model settings [e.g. a mix of open/closed source foundation models], or agentic systems



## **29. Breach By A Thousand Leaks: Unsafe Information Leakage in `Safe' AI Responses**

cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2407.02551v2) [paper-pdf](http://arxiv.org/pdf/2407.02551v2)

**Authors**: David Glukhov, Ziwen Han, Ilia Shumailov, Vardan Papyan, Nicolas Papernot

**Abstract**: Vulnerability of Frontier language models to misuse and jailbreaks has prompted the development of safety measures like filters and alignment training in an effort to ensure safety through robustness to adversarially crafted prompts. We assert that robustness is fundamentally insufficient for ensuring safety goals, and current defenses and evaluation methods fail to account for risks of dual-intent queries and their composition for malicious goals. To quantify these risks, we introduce a new safety evaluation framework based on impermissible information leakage of model outputs and demonstrate how our proposed question-decomposition attack can extract dangerous knowledge from a censored LLM more effectively than traditional jailbreaking. Underlying our proposed evaluation method is a novel information-theoretic threat model of inferential adversaries, distinguished from security adversaries, such as jailbreaks, in that success is measured by inferring impermissible knowledge from victim outputs as opposed to forcing explicitly impermissible outputs from the victim. Through our information-theoretic framework, we show that to ensure safety against inferential adversaries, defense mechanisms must ensure information censorship, bounding the leakage of impermissible information. However, we prove that such defenses inevitably incur a safety-utility trade-off.



## **30. FAIR-TAT: Improving Model Fairness Using Targeted Adversarial Training**

cs.LG

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23142v1) [paper-pdf](http://arxiv.org/pdf/2410.23142v1)

**Authors**: Tejaswini Medi, Steffen Jung, Margret Keuper

**Abstract**: Deep neural networks are susceptible to adversarial attacks and common corruptions, which undermine their robustness. In order to enhance model resilience against such challenges, Adversarial Training (AT) has emerged as a prominent solution. Nevertheless, adversarial robustness is often attained at the expense of model fairness during AT, i.e., disparity in class-wise robustness of the model. While distinctive classes become more robust towards such adversaries, hard to detect classes suffer. Recently, research has focused on improving model fairness specifically for perturbed images, overlooking the accuracy of the most likely non-perturbed data. Additionally, despite their robustness against the adversaries encountered during model training, state-of-the-art adversarial trained models have difficulty maintaining robustness and fairness when confronted with diverse adversarial threats or common corruptions. In this work, we address the above concerns by introducing a novel approach called Fair Targeted Adversarial Training (FAIR-TAT). We show that using targeted adversarial attacks for adversarial training (instead of untargeted attacks) can allow for more favorable trade-offs with respect to adversarial fairness. Empirical results validate the efficacy of our approach.



## **31. Reassessing Noise Augmentation Methods in the Context of Adversarial Speech**

eess.AS

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2409.01813v2) [paper-pdf](http://arxiv.org/pdf/2409.01813v2)

**Authors**: Karla Pizzi, Matías Pizarro, Asja Fischer

**Abstract**: In this study, we investigate if noise-augmented training can concurrently improve adversarial robustness in automatic speech recognition (ASR) systems. We conduct a comparative analysis of the adversarial robustness of four different state-of-the-art ASR architectures, where each of the ASR architectures is trained under three different augmentation conditions: one subject to background noise, speed variations, and reverberations, another subject to speed variations only, and a third without any form of data augmentation. The results demonstrate that noise augmentation not only improves model performance on noisy speech but also the model's robustness to adversarial attacks.



## **32. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2305.17000v7) [paper-pdf](http://arxiv.org/pdf/2305.17000v7)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **33. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23091v1) [paper-pdf](http://arxiv.org/pdf/2410.23091v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark).



## **34. Robustifying automatic speech recognition by extracting slowly varying features**

eess.AS

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2112.07400v2) [paper-pdf](http://arxiv.org/pdf/2112.07400v2)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: In the past few years, it has been shown that deep learning systems are highly vulnerable under attacks with adversarial examples. Neural-network-based automatic speech recognition (ASR) systems are no exception. Targeted and untargeted attacks can modify an audio input signal in such a way that humans still recognise the same words, while ASR systems are steered to predict a different transcription. In this paper, we propose a defense mechanism against targeted adversarial attacks consisting in removing fast-changing features from the audio signals, either by applying slow feature analysis, a low-pass filter, or both, before feeding the input to the ASR system. We perform an empirical analysis of hybrid ASR models trained on data pre-processed in such a way. While the resulting models perform quite well on benign data, they are significantly more robust against targeted adversarial attacks: Our final, proposed model shows a performance on clean data similar to the baseline model, while being more than four times more robust.



## **35. Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections**

cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.03052v2) [paper-pdf](http://arxiv.org/pdf/2406.03052v2)

**Authors**: Zihan Luo, Hong Huang, Yongkang Zhou, Jiping Zhang, Nuo Chen, Hai Jin

**Abstract**: Despite the remarkable capabilities demonstrated by Graph Neural Networks (GNNs) in graph-related tasks, recent research has revealed the fairness vulnerabilities in GNNs when facing malicious adversarial attacks. However, all existing fairness attacks require manipulating the connectivity between existing nodes, which may be prohibited in reality. To this end, we introduce a Node Injection-based Fairness Attack (NIFA), exploring the vulnerabilities of GNN fairness in such a more realistic setting. In detail, NIFA first designs two insightful principles for node injection operations, namely the uncertainty-maximization principle and homophily-increase principle, and then optimizes injected nodes' feature matrix to further ensure the effectiveness of fairness attacks. Comprehensive experiments on three real-world datasets consistently demonstrate that NIFA can significantly undermine the fairness of mainstream GNNs, even including fairness-aware GNNs, by injecting merely 1% of nodes. We sincerely hope that our work can stimulate increasing attention from researchers on the vulnerability of GNN fairness, and encourage the development of corresponding defense mechanisms. Our code and data are released at: https://github.com/CGCL-codes/NIFA.



## **36. Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector**

cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22888v1) [paper-pdf](http://arxiv.org/pdf/2410.22888v1)

**Authors**: Youcheng Huang, Fengbin Zhu, Jingkun Tang, Pan Zhou, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua

**Abstract**: Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature. To facilitate research on this critical safety problem, we first construct a new laRge-scale Adervsarial images dataset with Diverse hArmful Responses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses. With the new RADAR dataset, we further develop a novel and effective iN-time Embedding-based AdveRSarial Image DEtection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call the attacking direction, to achieve the detection of adversarial images against benign ones in the input. Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency, and cross-model transferrability of our proposed method. Our code is available at https://github.com/mob-scu/RADAR-NEARSIDE



## **37. Stealing User Prompts from Mixture of Experts**

cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22884v1) [paper-pdf](http://arxiv.org/pdf/2410.22884v1)

**Authors**: Itay Yona, Ilia Shumailov, Jamie Hayes, Nicholas Carlini

**Abstract**: Mixture-of-Experts (MoE) models improve the efficiency and scalability of dense language models by routing each token to a small number of experts in each layer. In this paper, we show how an adversary that can arrange for their queries to appear in the same batch of examples as a victim's queries can exploit Expert-Choice-Routing to fully disclose a victim's prompt. We successfully demonstrate the effectiveness of this attack on a two-layer Mixtral model, exploiting the tie-handling behavior of the torch.topk CUDA implementation. Our results show that we can extract the entire prompt using $O({VM}^2)$ queries (with vocabulary size $V$ and prompt length $M$) or 100 queries on average per token in the setting we consider. This is the first attack to exploit architectural flaws for the purpose of extracting user prompts, introducing a new class of LLM vulnerabilities.



## **38. Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation**

cs.IR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22844v1) [paper-pdf](http://arxiv.org/pdf/2410.22844v1)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Adversarial Collaborative Filtering (ACF), which typically applies adversarial perturbations at user and item embeddings through adversarial training, is widely recognized as an effective strategy for enhancing the robustness of Collaborative Filtering (CF) recommender systems against poisoning attacks. Besides, numerous studies have empirically shown that ACF can also improve recommendation performance compared to traditional CF. Despite these empirical successes, the theoretical understanding of ACF's effectiveness in terms of both performance and robustness remains unclear. To bridge this gap, in this paper, we first theoretically show that ACF can achieve a lower recommendation error compared to traditional CF with the same training epochs in both clean and poisoned data contexts. Furthermore, by establishing bounds for reductions in recommendation error during ACF's optimization process, we find that applying personalized magnitudes of perturbation for different users based on their embedding scales can further improve ACF's effectiveness. Building on these theoretical understandings, we propose Personalized Magnitude Adversarial Collaborative Filtering (PamaCF). Extensive experiments demonstrate that PamaCF effectively defends against various types of poisoning attacks while significantly enhancing recommendation performance.



## **39. Geometry Cloak: Preventing TGS-based 3D Reconstruction from Copyrighted Images**

cs.CV

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22705v1) [paper-pdf](http://arxiv.org/pdf/2410.22705v1)

**Authors**: Qi Song, Ziyuan Luo, Ka Chun Cheung, Simon See, Renjie Wan

**Abstract**: Single-view 3D reconstruction methods like Triplane Gaussian Splatting (TGS) have enabled high-quality 3D model generation from just a single image input within seconds. However, this capability raises concerns about potential misuse, where malicious users could exploit TGS to create unauthorized 3D models from copyrighted images. To prevent such infringement, we propose a novel image protection approach that embeds invisible geometry perturbations, termed "geometry cloaks", into images before supplying them to TGS. These carefully crafted perturbations encode a customized message that is revealed when TGS attempts 3D reconstructions of the cloaked image. Unlike conventional adversarial attacks that simply degrade output quality, our method forces TGS to fail the 3D reconstruction in a specific way - by generating an identifiable customized pattern that acts as a watermark. This watermark allows copyright holders to assert ownership over any attempted 3D reconstructions made from their protected images. Extensive experiments have verified the effectiveness of our geometry cloak. Our project is available at https://qsong2001.github.io/geometry_cloak.



## **40. Backdoor Attack Against Vision Transformers via Attention Gradient-Based Image Erosion**

cs.CV

Accepted by IEEE GLOBECOM 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22678v1) [paper-pdf](http://arxiv.org/pdf/2410.22678v1)

**Authors**: Ji Guo, Hongwei Li, Wenbo Jiang, Guoming Lu

**Abstract**: Vision Transformers (ViTs) have outperformed traditional Convolutional Neural Networks (CNN) across various computer vision tasks. However, akin to CNN, ViTs are vulnerable to backdoor attacks, where the adversary embeds the backdoor into the victim model, causing it to make wrong predictions about testing samples containing a specific trigger. Existing backdoor attacks against ViTs have the limitation of failing to strike an optimal balance between attack stealthiness and attack effectiveness.   In this work, we propose an Attention Gradient-based Erosion Backdoor (AGEB) targeted at ViTs. Considering the attention mechanism of ViTs, AGEB selectively erodes pixels in areas of maximal attention gradient, embedding a covert backdoor trigger. Unlike previous backdoor attacks against ViTs, AGEB achieves an optimal balance between attack stealthiness and attack effectiveness, ensuring the trigger remains invisible to human detection while preserving the model's accuracy on clean samples. Extensive experimental evaluations across various ViT architectures and datasets confirm the effectiveness of AGEB, achieving a remarkable Attack Success Rate (ASR) without diminishing Clean Data Accuracy (CDA). Furthermore, the stealthiness of AGEB is rigorously validated, demonstrating minimal visual discrepancies between the clean and the triggered images.



## **41. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

cs.SE

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22663v1) [paper-pdf](http://arxiv.org/pdf/2410.22663v1)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains, such as toxicity detection, chatbot consulting, and review analysis. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Several studies indicate that traditional metrics, such as model confidence and accuracy, are insufficient to build human trust in ML models. These models often learn spurious correlations during training and predict based on them during inference. In the real world, where such correlations are absent, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods, which is time-consuming and not scalable. We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers, which automatically checks whether the prediction-contributing words are related to the predicted class using explanation methods and word embeddings. To demonstrate its practical usefulness, we introduce a novel adversarial attack method targeting trustworthiness issues identified by TOKI. We compare TOKI with a naive baseline based solely on model confidence using human-created ground truths of 6,000 predictions. We also compare TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided adversarial attack method is more effective with fewer perturbations than A2T.



## **42. AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents**

cs.CR

15 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.17401v2) [paper-pdf](http://arxiv.org/pdf/2410.17401v2)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Vision Language Models (VLMs) have revolutionized the creation of generalist web agents, empowering them to autonomously complete diverse tasks on real-world websites, thereby boosting human efficiency and productivity. However, despite their remarkable capabilities, the safety and security of these agents against malicious attacks remain critically underexplored, raising significant concerns about their safe deployment. To uncover and exploit such vulnerabilities in web agents, we provide AdvWeb, a novel black-box attack framework designed against web agents. AdvWeb trains an adversarial prompter model that generates and injects adversarial prompts into web pages, misleading web agents into executing targeted adversarial actions such as inappropriate stock purchases or incorrect bank transactions, actions that could lead to severe real-world consequences. With only black-box access to the web agent, we train and optimize the adversarial prompter model using DPO, leveraging both successful and failed attack strings against the target agent. Unlike prior approaches, our adversarial string injection maintains stealth and control: (1) the appearance of the website remains unchanged before and after the attack, making it nearly impossible for users to detect tampering, and (2) attackers can modify specific substrings within the generated adversarial string to seamlessly change the attack objective (e.g., purchasing stocks from a different company), enhancing attack flexibility and efficiency. We conduct extensive evaluations, demonstrating that AdvWeb achieves high success rates in attacking SOTA GPT-4V-based VLM agent across various web tasks. Our findings expose critical vulnerabilities in current LLM/VLM-based agents, emphasizing the urgent need for developing more reliable web agents and effective defenses. Our code and data are available at https://ai-secure.github.io/AdvWeb/ .



## **43. LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate**

cs.CV

NeurIPS 2024 Camera Ready

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2405.13985v2) [paper-pdf](http://arxiv.org/pdf/2405.13985v2)

**Authors**: Anthony Fuller, Daniel G. Kyrollos, Yousef Yassin, James R. Green

**Abstract**: High-resolution images offer more information about scenes that can improve model accuracy. However, the dominant model architecture in computer vision, the vision transformer (ViT), cannot effectively leverage larger images without finetuning -- ViTs poorly extrapolate to more patches at test time, although transformers offer sequence length flexibility. We attribute this shortcoming to the current patch position encoding methods, which create a distribution shift when extrapolating.   We propose a drop-in replacement for the position encoding of plain ViTs that restricts attention heads to fixed fields of view, pointed in different directions, using 2D attention masks. Our novel method, called LookHere, provides translation-equivariance, ensures attention head diversity, and limits the distribution shift that attention heads face when extrapolating. We demonstrate that LookHere improves performance on classification (avg. 1.6%), against adversarial attack (avg. 5.4%), and decreases calibration error (avg. 1.5%) -- on ImageNet without extrapolation. With extrapolation, LookHere outperforms the current SoTA position encoding method, 2D-RoPE, by 21.7% on ImageNet when trained at $224^2$ px and tested at $1024^2$ px. Additionally, we release a high-resolution test set to improve the evaluation of high-resolution image classifiers, called ImageNet-HR.



## **44. Power side-channel leakage localization through adversarial training of deep neural networks**

cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22425v1) [paper-pdf](http://arxiv.org/pdf/2410.22425v1)

**Authors**: Jimmy Gammell, Anand Raghunathan, Kaushik Roy

**Abstract**: Supervised deep learning has emerged as an effective tool for carrying out power side-channel attacks on cryptographic implementations. While increasingly-powerful deep learning-based attacks are regularly published, comparatively-little work has gone into using deep learning to defend against these attacks. In this work we propose a technique for identifying which timesteps in a power trace are responsible for leaking a cryptographic key, through an adversarial game between a deep learning-based side-channel attacker which seeks to classify a sensitive variable from the power traces recorded during encryption, and a trainable noise generator which seeks to thwart this attack by introducing a minimal amount of noise into the power traces. We demonstrate on synthetic datasets that our method can outperform existing techniques in the presence of common countermeasures such as Boolean masking and trace desynchronization. Results on real datasets are weak because the technique is highly sensitive to hyperparameters and early-stop point, and we lack a holdout dataset with ground truth knowledge of leaking points for model selection. Nonetheless, we believe our work represents an important first step towards deep side-channel leakage localization without relying on strong assumptions about the implementation or the nature of its leakage. An open-source PyTorch implementation of our experiments is provided.



## **45. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

cs.LG

20 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22307v1) [paper-pdf](http://arxiv.org/pdf/2410.22307v1)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: Open-source Large Language Models (LLMs) have recently demonstrated remarkable capabilities in natural language understanding and generation, leading to widespread adoption across various domains. However, their increasing model sizes render local deployment impractical for individual users, pushing many to rely on computing service providers for inference through a blackbox API. This reliance introduces a new risk: a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby delivering inferior outputs while benefiting from cost savings. In this paper, we formalize the problem of verifiable inference for LLMs. Existing verifiable computing solutions based on cryptographic or game-theoretic techniques are either computationally uneconomical or rest on strong assumptions. We introduce SVIP, a secret-based verifiable LLM inference protocol that leverages intermediate outputs from LLM as unique model identifiers. By training a proxy task on these outputs and requiring the computing provider to return both the generated text and the processed intermediate outputs, users can reliably verify whether the computing provider is acting honestly. In addition, the integration of a secret mechanism further enhances the security of our protocol. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per query for verification.



## **46. Embedding-based classifiers can detect prompt injection attacks**

cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22284v1) [paper-pdf](http://arxiv.org/pdf/2410.22284v1)

**Authors**: Md. Ahsan Ayub, Subhabrata Majumdar

**Abstract**: Large Language Models (LLMs) are seeing significant adoption in every type of organization due to their exceptional generative capabilities. However, LLMs are found to be vulnerable to various adversarial attacks, particularly prompt injection attacks, which trick them into producing harmful or inappropriate content. Adversaries execute such attacks by crafting malicious prompts to deceive the LLMs. In this paper, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against this severe threat. We leverage three commonly used embedding models to generate embeddings of malicious and benign prompts and utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in open-source implementations, which use encoder-only neural networks.



## **47. AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts**

cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22143v1) [paper-pdf](http://arxiv.org/pdf/2410.22143v1)

**Authors**: Vishal Kumar, Zeyi Liao, Jaylen Jones, Huan Sun

**Abstract**: Although large language models (LLMs) are typically aligned, they remain vulnerable to jailbreaking through either carefully crafted prompts in natural language or, interestingly, gibberish adversarial suffixes. However, gibberish tokens have received relatively less attention despite their success in attacking aligned LLMs. Recent work, AmpleGCG~\citep{liao2024amplegcg}, demonstrates that a generative model can quickly produce numerous customizable gibberish adversarial suffixes for any harmful query, exposing a range of alignment gaps in out-of-distribution (OOD) language spaces. To bring more attention to this area, we introduce AmpleGCG-Plus, an enhanced version that achieves better performance in fewer attempts. Through a series of exploratory experiments, we identify several training strategies to improve the learning of gibberish suffixes. Our results, verified under a strict evaluation setting, show that it outperforms AmpleGCG on both open-weight and closed-source models, achieving increases in attack success rate (ASR) of up to 17\% in the white-box setting against Llama-2-7B-chat, and more than tripling ASR in the black-box setting against GPT-4. Notably, AmpleGCG-Plus jailbreaks the newer GPT-4o series of models at similar rates to GPT-4, and, uncovers vulnerabilities against the recently proposed circuit breakers defense. We publicly release AmpleGCG-Plus along with our collected training datasets.



## **48. Iterative Window Mean Filter: Thwarting Diffusion-based Adversarial Purification**

cs.CR

Accepted in IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2408.10673v3) [paper-pdf](http://arxiv.org/pdf/2408.10673v3)

**Authors**: Hanrui Wang, Ruoxi Sun, Cunjian Chen, Minhui Xue, Lay-Ki Soon, Shuo Wang, Zhe Jin

**Abstract**: Face authentication systems have brought significant convenience and advanced developments, yet they have become unreliable due to their sensitivity to inconspicuous perturbations, such as adversarial attacks. Existing defenses often exhibit weaknesses when facing various attack algorithms and adaptive attacks or compromise accuracy for enhanced security. To address these challenges, we have developed a novel and highly efficient non-deep-learning-based image filter called the Iterative Window Mean Filter (IWMF) and proposed a new framework for adversarial purification, named IWMF-Diff, which integrates IWMF and denoising diffusion models. These methods can function as pre-processing modules to eliminate adversarial perturbations without necessitating further modifications or retraining of the target system. We demonstrate that our proposed methodologies fulfill four critical requirements: preserved accuracy, improved security, generalizability to various threats in different settings, and better resistance to adaptive attacks. This performance surpasses that of the state-of-the-art adversarial purification method, DiffPure.



## **49. Forging the Forger: An Attempt to Improve Authorship Verification via Data Augmentation**

cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2403.11265v2) [paper-pdf](http://arxiv.org/pdf/2403.11265v2)

**Authors**: Silvia Corbara, Alejandro Moreo

**Abstract**: Authorship Verification (AV) is a text classification task concerned with inferring whether a candidate text has been written by one specific author or by someone else. It has been shown that many AV systems are vulnerable to adversarial attacks, where a malicious author actively tries to fool the classifier by either concealing their writing style, or by imitating the style of another author. In this paper, we investigate the potential benefits of augmenting the classifier training set with (negative) synthetic examples. These synthetic examples are generated to imitate the style of the author of interest. We analyze the improvements in classifier prediction that this augmentation brings to bear in the task of AV in an adversarial setting. In particular, we experiment with three different generator architectures (one based on Recurrent Neural Networks, another based on small-scale transformers, and another based on the popular GPT model) and with two training strategies (one inspired by standard Language Models, and another inspired by Wasserstein Generative Adversarial Networks). We evaluate our hypothesis on five datasets (three of which have been specifically collected to represent an adversarial setting) and using two learning algorithms for the AV classifier (Support Vector Machines and Convolutional Neural Networks). This experimentation has yielded negative results, revealing that, although our methodology proves effective in many adversarial settings, its benefits are too sporadic for a pragmatical application.



## **50. On the Robustness of Adversarial Training Against Uncertainty Attacks**

cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21952v1) [paper-pdf](http://arxiv.org/pdf/2410.21952v1)

**Authors**: Emanuele Ledda, Giovanni Scodeller, Daniele Angioni, Giorgio Piras, Antonio Emanuele Cinà, Giorgio Fumera, Battista Biggio, Fabio Roli

**Abstract**: In learning problems, the noise inherent to the task at hand hinders the possibility to infer without a certain degree of uncertainty. Quantifying this uncertainty, regardless of its wide use, assumes high relevance for security-sensitive applications. Within these scenarios, it becomes fundamental to guarantee good (i.e., trustworthy) uncertainty measures, which downstream modules can securely employ to drive the final decision-making process. However, an attacker may be interested in forcing the system to produce either (i) highly uncertain outputs jeopardizing the system's availability or (ii) low uncertainty estimates, making the system accept uncertain samples that would instead require a careful inspection (e.g., human intervention). Therefore, it becomes fundamental to understand how to obtain robust uncertainty estimates against these kinds of attacks. In this work, we reveal both empirically and theoretically that defending against adversarial examples, i.e., carefully perturbed samples that cause misclassification, additionally guarantees a more secure, trustworthy uncertainty estimate under common attack scenarios without the need for an ad-hoc defense strategy. To support our claims, we evaluate multiple adversarial-robust models from the publicly available benchmark RobustBench on the CIFAR-10 and ImageNet datasets.



