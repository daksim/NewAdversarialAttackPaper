# Latest Adversarial Attack Papers
**update at 2023-05-23 19:29:05**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. And/or trade-off in artificial neurons: impact on adversarial robustness**

cs.LG

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2102.07389v3) [paper-pdf](http://arxiv.org/pdf/2102.07389v3)

**Authors**: Alessandro Fontana

**Abstract**: Despite the success of neural networks, the issue of classification robustness remains, particularly highlighted by adversarial examples. In this paper, we address this challenge by focusing on the continuum of functions implemented in artificial neurons, ranging from pure AND gates to pure OR gates. Our hypothesis is that the presence of a sufficient number of OR-like neurons in a network can lead to classification brittleness and increased vulnerability to adversarial attacks. We define AND-like neurons and propose measures to increase their proportion in the network. These measures involve rescaling inputs to the [-1,1] interval and reducing the number of points in the steepest section of the sigmoidal activation function. A crucial component of our method is the comparison between a neuron's output distribution when fed with the actual dataset and a randomised version called the "scrambled dataset." Experimental results on the MNIST dataset suggest that our approach holds promise as a direction for further exploration.



## **2. Analyzing the Shuffle Model through the Lens of Quantitative Information Flow**

cs.CR

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.13075v1) [paper-pdf](http://arxiv.org/pdf/2305.13075v1)

**Authors**: Mireya Jurado, Ramon G. Gonze, Mário S. Alvim, Catuscia Palamidessi

**Abstract**: Local differential privacy (LDP) is a variant of differential privacy (DP) that avoids the need for a trusted central curator, at the cost of a worse trade-off between privacy and utility. The shuffle model is a way to provide greater anonymity to users by randomly permuting their messages, so that the link between users and their reported values is lost to the data collector. By combining an LDP mechanism with a shuffler, privacy can be improved at no cost for the accuracy of operations insensitive to permutations, thereby improving utility in many tasks. However, the privacy implications of shuffling are not always immediately evident, and derivations of privacy bounds are made on a case-by-case basis.   In this paper, we analyze the combination of LDP with shuffling in the rigorous framework of quantitative information flow (QIF), and reason about the resulting resilience to inference attacks. QIF naturally captures randomization mechanisms as information-theoretic channels, thus allowing for precise modeling of a variety of inference attacks in a natural way and for measuring the leakage of private information under these attacks. We exploit symmetries of the particular combination of k-RR mechanisms with the shuffle model to achieve closed formulas that express leakage exactly. In particular, we provide formulas that show how shuffling improves protection against leaks in the local model, and study how leakage behaves for various values of the privacy parameter of the LDP mechanism.   In contrast to the strong adversary from differential privacy, we focus on an uninformed adversary, who does not know the value of any individual in the dataset. This adversary is often more realistic as a consumer of statistical datasets, and we show that in some situations mechanisms that are equivalent w.r.t. the strong adversary can provide different privacy guarantees under the uninformed one.



## **3. Latent Magic: An Investigation into Adversarial Examples Crafted in the Semantic Latent Space**

cs.LG

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12906v1) [paper-pdf](http://arxiv.org/pdf/2305.12906v1)

**Authors**: BoYang Zheng

**Abstract**: Adversarial attacks against Deep Neural Networks(DNN) have been a crutial topic ever since \cite{goodfellow} purposed the vulnerability of DNNs. However, most prior works craft adversarial examples in the pixel space, following the $l_p$ norm constraint. In this paper, we give intuitional explain about why crafting adversarial examples in the latent space is equally efficient and important. We purpose a framework for crafting adversarial examples in semantic latent space based on an pre-trained Variational Auto Encoder from state-of-art Stable Diffusion Model\cite{SDM}. We also show that adversarial examples crafted in the latent space can also achieve a high level of fool rate. However, examples crafted from latent space are often hard to evaluated, as they doesn't follow a certain $l_p$ norm constraint, which is a big challenge for existing researches. To efficiently and accurately evaluate the adversarial examples crafted in the latent space, we purpose \textbf{a novel evaluation matric} based on SSIM\cite{SSIM} loss and fool rate.Additionally, we explain why FID\cite{FID} is not suitable for measuring such adversarial examples. To the best of our knowledge, it's the first evaluation metrics that is specifically designed to evaluate the quality of a adversarial attack. We also investigate the transferability of adversarial examples crafted in the latent space and show that they have superiority over adversarial examples crafted in the pixel space.



## **4. Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game**

cs.GT

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12872v1) [paper-pdf](http://arxiv.org/pdf/2305.12872v1)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Xini Yu, Jiakai Wang, Aishan Liu, Yaodong Yang, Xianglong Liu

**Abstract**: In this study, we explore the robustness of cooperative multi-agent reinforcement learning (c-MARL) against Byzantine failures, where any agent can enact arbitrary, worst-case actions due to malfunction or adversarial attack. To address the uncertainty that any agent can be adversarial, we propose a Bayesian Adversarial Robust Dec-POMDP (BARDec-POMDP) framework, which views Byzantine adversaries as nature-dictated types, represented by a separate transition. This allows agents to learn policies grounded on their posterior beliefs about the type of other agents, fostering collaboration with identified allies and minimizing vulnerability to adversarial manipulation. We define the optimal solution to the BARDec-POMDP as an ex post robust Bayesian Markov perfect equilibrium, which we proof to exist and weakly dominates the equilibrium of previous robust MARL approaches. To realize this equilibrium, we put forward a two-timescale actor-critic algorithm with almost sure convergence under specific conditions. Experimentation on matrix games, level-based foraging and StarCraft II indicate that, even under worst-case perturbations, our method successfully acquires intricate micromanagement skills and adaptively aligns with allies, demonstrating resilience against non-oblivious adversaries, random allies, observation-based attacks, and transfer-based attacks.



## **5. Towards Benchmarking and Assessing Visual Naturalness of Physical World Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12863v1) [paper-pdf](http://arxiv.org/pdf/2305.12863v1)

**Authors**: Simin Li, Shuing Zhang, Gujun Chen, Dong Wang, Pu Feng, Jiakai Wang, Aishan Liu, Xin Yi, Xianglong Liu

**Abstract**: Physical world adversarial attack is a highly practical and threatening attack, which fools real world deep learning systems by generating conspicuous and maliciously crafted real world artifacts. In physical world attacks, evaluating naturalness is highly emphasized since human can easily detect and remove unnatural attacks. However, current studies evaluate naturalness in a case-by-case fashion, which suffers from errors, bias and inconsistencies. In this paper, we take the first step to benchmark and assess visual naturalness of physical world attacks, taking autonomous driving scenario as the first attempt. First, to benchmark attack naturalness, we contribute the first Physical Attack Naturalness (PAN) dataset with human rating and gaze. PAN verifies several insights for the first time: naturalness is (disparately) affected by contextual features (i.e., environmental and semantic variations) and correlates with behavioral feature (i.e., gaze signal). Second, to automatically assess attack naturalness that aligns with human ratings, we further introduce Dual Prior Alignment (DPA) network, which aims to embed human knowledge into model reasoning process. Specifically, DPA imitates human reasoning in naturalness assessment by rating prior alignment and mimics human gaze behavior by attentive prior alignment. We hope our work fosters researches to improve and automatically assess naturalness of physical world attacks. Our code and dataset can be found at https://github.com/zhangsn-19/PAN.



## **6. Flying Adversarial Patches: Manipulating the Behavior of Deep Learning-based Autonomous Multirotors**

cs.RO

6 pages, 5 figures, Workshop on Multi-Robot Learning, International  Conference on Robotics and Automation (ICRA)

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12859v1) [paper-pdf](http://arxiv.org/pdf/2305.12859v1)

**Authors**: Pia Hanfeld, Marina M. -C. Höhne, Michael Bussmann, Wolfgang Hönig

**Abstract**: Autonomous flying robots, e.g. multirotors, often rely on a neural network that makes predictions based on a camera image. These deep learning (DL) models can compute surprising results if applied to input images outside the training domain. Adversarial attacks exploit this fault, for example, by computing small images, so-called adversarial patches, that can be placed in the environment to manipulate the neural network's prediction. We introduce flying adversarial patches, where an image is mounted on another flying robot and therefore can be placed anywhere in the field of view of a victim multirotor. For an effective attack, we compare three methods that simultaneously optimize the adversarial patch and its position in the input image. We perform an empirical validation on a publicly available DL model and dataset for autonomous multirotors. Ultimately, our attacking multirotor would be able to gain full control over the motions of the victim multirotor.



## **7. Uncertainty-based Detection of Adversarial Attacks in Semantic Segmentation**

cs.CV

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12825v1) [paper-pdf](http://arxiv.org/pdf/2305.12825v1)

**Authors**: Kira Maag, Asja Fischer

**Abstract**: State-of-the-art deep neural networks have proven to be highly powerful in a broad range of tasks, including semantic image segmentation. However, these networks are vulnerable against adversarial attacks, i.e., non-perceptible perturbations added to the input image causing incorrect predictions, which is hazardous in safety-critical applications like automated driving. Adversarial examples and defense strategies are well studied for the image classification task, while there has been limited research in the context of semantic segmentation. First works however show that the segmentation outcome can be severely distorted by adversarial attacks. In this work, we introduce an uncertainty-based method for the detection of adversarial attacks in semantic segmentation. We observe that uncertainty as for example captured by the entropy of the output distribution behaves differently on clean and perturbed images using this property to distinguish between the two cases. Our method works in a light-weight and post-processing manner, i.e., we do not modify the model or need knowledge of the process used for generating adversarial examples. In a thorough empirical analysis, we demonstrate the ability of our approach to detect perturbed images across multiple types of adversarial attacks.



## **8. The defender's perspective on automatic speaker verification: An overview**

cs.SD

Submitted to IJCAI 2023 Workshop

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12804v1) [paper-pdf](http://arxiv.org/pdf/2305.12804v1)

**Authors**: Haibin Wu, Jiawen Kang, Lingwei Meng, Helen Meng, Hung-yi Lee

**Abstract**: Automatic speaker verification (ASV) plays a critical role in security-sensitive environments. Regrettably, the reliability of ASV has been undermined by the emergence of spoofing attacks, such as replay and synthetic speech, as well as adversarial attacks and the relatively new partially fake speech. While there are several review papers that cover replay and synthetic speech, and adversarial attacks, there is a notable gap in a comprehensive review that addresses defense against adversarial attacks and the recently emerged partially fake speech. Thus, the aim of this paper is to provide a thorough and systematic overview of the defense methods used against these types of attacks.



## **9. FGAM:Fast Adversarial Malware Generation Method Based on Gradient Sign**

cs.CR

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.12770v1) [paper-pdf](http://arxiv.org/pdf/2305.12770v1)

**Authors**: Kun Li, Fan Zhang, Wei Guo

**Abstract**: Malware detection models based on deep learning have been widely used, but recent research shows that deep learning models are vulnerable to adversarial attacks. Adversarial attacks are to deceive the deep learning model by generating adversarial samples. When adversarial attacks are performed on the malware detection model, the attacker will generate adversarial malware with the same malicious functions as the malware, and make the detection model classify it as benign software. Studying adversarial malware generation can help model designers improve the robustness of malware detection models. At present, in the work on adversarial malware generation for byte-to-image malware detection models, there are mainly problems such as large amount of injection perturbation and low generation efficiency. Therefore, this paper proposes FGAM (Fast Generate Adversarial Malware), a method for fast generating adversarial malware, which iterates perturbed bytes according to the gradient sign to enhance adversarial capability of the perturbed bytes until the adversarial malware is successfully generated. It is experimentally verified that the success rate of the adversarial malware deception model generated by FGAM is increased by about 84\% compared with existing methods.



## **10. On The Empirical Effectiveness of Unrealistic Adversarial Hardening Against Realistic Adversarial Attacks**

cs.LG

S&P 2023

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2202.03277v2) [paper-pdf](http://arxiv.org/pdf/2202.03277v2)

**Authors**: Salijona Dyrmishi, Salah Ghamizi, Thibault Simonetto, Yves Le Traon, Maxime Cordy

**Abstract**: While the literature on security attacks and defense of Machine Learning (ML) systems mostly focuses on unrealistic adversarial examples, recent research has raised concern about the under-explored field of realistic adversarial attacks and their implications on the robustness of real-world systems. Our paper paves the way for a better understanding of adversarial robustness against realistic attacks and makes two major contributions. First, we conduct a study on three real-world use cases (text classification, botnet detection, malware detection)) and five datasets in order to evaluate whether unrealistic adversarial examples can be used to protect models against realistic examples. Our results reveal discrepancies across the use cases, where unrealistic examples can either be as effective as the realistic ones or may offer only limited improvement. Second, to explain these results, we analyze the latent representation of the adversarial examples generated with realistic and unrealistic attacks. We shed light on the patterns that discriminate which unrealistic examples can be used for effective hardening. We release our code, datasets and models to support future research in exploring how to reduce the gap between unrealistic and realistic adversarial attacks.



## **11. RAIN: RegulArization on Input and Network for Black-Box Domain Adaptation**

cs.CV

Accepted by IJCAI 2023

**SubmitDate**: 2023-05-21    [abs](http://arxiv.org/abs/2208.10531v3) [paper-pdf](http://arxiv.org/pdf/2208.10531v3)

**Authors**: Qucheng Peng, Zhengming Ding, Lingjuan Lyu, Lichao Sun, Chen Chen

**Abstract**: Source-Free domain adaptation transits the source-trained model towards target domain without exposing the source data, trying to dispel these concerns about data privacy and security. However, this paradigm is still at risk of data leakage due to adversarial attacks on the source model. Hence, the Black-Box setting only allows to use the outputs of source model, but still suffers from overfitting on the source domain more severely due to source model's unseen weights. In this paper, we propose a novel approach named RAIN (RegulArization on Input and Network) for Black-Box domain adaptation from both input-level and network-level regularization. For the input-level, we design a new data augmentation technique as Phase MixUp, which highlights task-relevant objects in the interpolations, thus enhancing input-level regularization and class consistency for target models. For network-level, we develop a Subnetwork Distillation mechanism to transfer knowledge from the target subnetwork to the full target network via knowledge distillation, which thus alleviates overfitting on the source domain by learning diverse target representations. Extensive experiments show that our method achieves state-of-the-art performance on several cross-domain benchmarks under both single- and multi-source black-box domain adaptation.



## **12. Dynamic Transformers Provide a False Sense of Efficiency**

cs.CL

Accepted by ACL2023

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.12228v1) [paper-pdf](http://arxiv.org/pdf/2305.12228v1)

**Authors**: Yiming Chen, Simin Chen, Zexin Li, Wei Yang, Cong Liu, Robby T. Tan, Haizhou Li

**Abstract**: Despite much success in natural language processing (NLP), pre-trained language models typically lead to a high computational cost during inference. Multi-exit is a mainstream approach to address this issue by making a trade-off between efficiency and accuracy, where the saving of computation comes from an early exit. However, whether such saving from early-exiting is robust remains unknown. Motivated by this, we first show that directly adapting existing adversarial attack approaches targeting model accuracy cannot significantly reduce inference efficiency. To this end, we propose a simple yet effective attacking framework, SAME, a novel slowdown attack framework on multi-exit models, which is specially tailored to reduce the efficiency of the multi-exit models. By leveraging the multi-exit models' design characteristics, we utilize all internal predictions to guide the adversarial sample generation instead of merely considering the final prediction. Experiments on the GLUE benchmark show that SAME can effectively diminish the efficiency gain of various multi-exit models by 80% on average, convincingly validating its effectiveness and generalization ability.



## **13. RNNS: Representation Nearest Neighbor Search Black-Box Attack on Code Models**

cs.CR

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.05896v2) [paper-pdf](http://arxiv.org/pdf/2305.05896v2)

**Authors**: Jie Zhang, Wei Ma, Qiang Hu, Xiaofei Xie, Yves Le Traon, Yang Liu

**Abstract**: Pre-trained code models are mainly evaluated using the in-distribution test data. The robustness of models, i.e., the ability to handle hard unseen data, still lacks evaluation. In this paper, we propose a novel search-based black-box adversarial attack guided by model behaviours for pre-trained programming language models, named Representation Nearest Neighbor Search(RNNS), to evaluate the robustness of Pre-trained PL models. Unlike other black-box adversarial attacks, RNNS uses the model-change signal to guide the search in the space of the variable names collected from real-world projects. Specifically, RNNS contains two main steps, 1) indicate which variable (attack position location) we should attack based on model uncertainty, and 2) search which adversarial tokens we should use for variable renaming according to the model behaviour observations. We evaluate RNNS on 6 code tasks (e.g., clone detection), 3 programming languages (Java, Python, and C), and 3 pre-trained code models: CodeBERT, GraphCodeBERT, and CodeT5. The results demonstrate that RNNS outperforms the state-of-the-art black-box attacking methods (MHM and ALERT) in terms of attack success rate (ASR) and query times (QT). The perturbation of generated adversarial examples from RNNS is smaller than the baselines with respect to the number of replaced variables and the variable length change. Our experiments also show that RNNS is efficient in attacking the defended models and is useful for adversarial training.



## **14. Towards Adversarially Robust Recommendation from Adaptive Fraudster Detection**

cs.IR

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2211.11534v3) [paper-pdf](http://arxiv.org/pdf/2211.11534v3)

**Authors**: Yuni Lai, Yulin Zhu, Wenqi Fan, Xiaoge Zhang, Kai Zhou

**Abstract**: The robustness of recommender systems under node injection attacks has garnered significant attention. Recently, GraphRfi, a GNN-based recommender system, was proposed and shown to effectively mitigate the impact of injected fake users. However, we demonstrate that GraphRfi remains vulnerable to attacks due to the supervised nature of its fraudster detection component, where obtaining clean labels is challenging in practice. In particular, we propose a powerful poisoning attack, MetaC, against both GNN-based and MF-based recommender systems. Furthermore, we analyze why GraphRfi fails under such an attack. Then, based on our insights obtained from vulnerability analysis, we design an adaptive fraudster detection module that explicitly considers label uncertainty. This module can serve as a plug-in for different recommender systems, resulting in a robust framework named PDR. Comprehensive experiments show that our defense approach outperforms other benchmark methods under attacks. Overall, our research presents an effective framework for integrating fraudster detection into recommendation systems to achieve adversarial robustness.



## **15. Annealing Self-Distillation Rectification Improves Adversarial Training**

cs.LG

10 pages + Appendix

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.12118v1) [paper-pdf](http://arxiv.org/pdf/2305.12118v1)

**Authors**: Yu-Yu Wu, Hung-Jui Wang, Shang-Tse Chen

**Abstract**: In standard adversarial training, models are optimized to fit one-hot labels within allowable adversarial perturbation budgets. However, the ignorance of underlying distribution shifts brought by perturbations causes the problem of robust overfitting. To address this issue and enhance adversarial robustness, we analyze the characteristics of robust models and identify that robust models tend to produce smoother and well-calibrated outputs. Based on the observation, we propose a simple yet effective method, Annealing Self-Distillation Rectification (ADR), which generates soft labels as a better guidance mechanism that accurately reflects the distribution shift under attack during adversarial training. By utilizing ADR, we can obtain rectified distributions that significantly improve model robustness without the need for pre-trained models or extensive extra computation. Moreover, our method facilitates seamless plug-and-play integration with other adversarial training techniques by replacing the hard labels in their objectives. We demonstrate the efficacy of ADR through extensive experiments and strong performances across datasets.



## **16. SneakyPrompt: Evaluating Robustness of Text-to-image Generative Models' Safety Filters**

cs.LG

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.12082v1) [paper-pdf](http://arxiv.org/pdf/2305.12082v1)

**Authors**: Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao

**Abstract**: Text-to-image generative models such as Stable Diffusion and DALL$\cdot$E 2 have attracted much attention since their publication due to their wide application in the real world. One challenging problem of text-to-image generative models is the generation of Not-Safe-for-Work (NSFW) content, e.g., those related to violence and adult. Therefore, a common practice is to deploy a so-called safety filter, which blocks NSFW content based on either text or image features. Prior works have studied the possible bypass of such safety filters. However, existing works are largely manual and specific to Stable Diffusion's official safety filter. Moreover, the bypass ratio of Stable Diffusion's safety filter is as low as 23.51% based on our evaluation.   In this paper, we propose the first automated attack framework, called SneakyPrompt, to evaluate the robustness of real-world safety filters in state-of-the-art text-to-image generative models. Our key insight is to search for alternative tokens in a prompt that generates NSFW images so that the generated prompt (called an adversarial prompt) bypasses existing safety filters. Specifically, SneakyPrompt utilizes reinforcement learning (RL) to guide an agent with positive rewards on semantic similarity and bypass success.   Our evaluation shows that SneakyPrompt successfully generated NSFW content using an online model DALL$\cdot$E 2 with its default, closed-box safety filter enabled. At the same time, we also deploy several open-source state-of-the-art safety filters on a Stable Diffusion model and show that SneakyPrompt not only successfully generates NSFW content, but also outperforms existing adversarial attacks in terms of the number of queries and image qualities.



## **17. STDLens: Model Hijacking-Resilient Federated Learning for Object Detection**

cs.CR

CVPR 2023. Source Code: https://github.com/git-disl/STDLens

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2303.11511v3) [paper-pdf](http://arxiv.org/pdf/2303.11511v3)

**Authors**: Ka-Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu

**Abstract**: Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates.



## **18. Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models**

cs.LG

19 pages, 5 figures

**SubmitDate**: 2023-05-20    [abs](http://arxiv.org/abs/2305.12066v1) [paper-pdf](http://arxiv.org/pdf/2305.12066v1)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-task learning (MTL) creates a single machine learning model called multi-task model to simultaneously perform multiple tasks. Although the security of single task classifiers has been extensively studied, there are several critical security research questions for multi-task models including 1) How secure are multi-task models to single task adversarial machine learning attacks, 2) Can adversarial attacks be designed to attack multiple tasks simultaneously, and 3) Does task sharing and adversarial training increase multi-task model robustness to adversarial attacks? In this paper, we answer these questions through careful analysis and rigorous experimentation. First, we develop na\"ive adaptation of single-task white-box attacks and analyze their inherent drawbacks. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking a multi-task model as an optimization problem based on averaged relative loss change, which can be solved by approximating the problem as an integer linear programming problem. Extensive evaluation on two popular MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrates the effectiveness of DGBA compared to na\"ive multi-task attack baselines on both clean and adversarially trained multi-task models. The results also reveal a fundamental trade-off between improving task accuracy by sharing parameters across tasks and undermining model robustness due to increased attack transferability from parameter sharing.



## **19. DAP: A Dynamic Adversarial Patch for Evading Person Detectors**

cs.CR

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2305.11618v1) [paper-pdf](http://arxiv.org/pdf/2305.11618v1)

**Authors**: Amira Guesmi, Ruitian Ding, Muhammad Abdullah Hanif, Ihsen Alouani, Muhammad Shafique

**Abstract**: In this paper, we present a novel approach for generating naturalistic adversarial patches without using GANs. Our proposed approach generates a Dynamic Adversarial Patch (DAP) that looks naturalistic while maintaining high attack efficiency and robustness in real-world scenarios. To achieve this, we redefine the optimization problem by introducing a new objective function, where a similarity metric is used to construct a similarity loss. This guides the patch to follow predefined patterns while maximizing the victim model's loss function. Our technique is based on directly modifying the pixel values in the patch which gives higher flexibility and larger space to incorporate multiple transformations compared to the GAN-based techniques. Furthermore, most clothing-based physical attacks assume static objects and ignore the possible transformations caused by non-rigid deformation due to changes in a person's pose. To address this limitation, we incorporate a ``Creases Transformation'' (CT) block, i.e., a preprocessing block following an Expectation Over Transformation (EOT) block used to generate a large variation of transformed patches incorporated in the training process to increase its robustness to different possible real-world distortions (e.g., creases in the clothing, rotation, re-scaling, random noise, brightness and contrast variations, etc.). We demonstrate that the presence of different real-world variations in clothing and object poses (i.e., above-mentioned distortions) lead to a drop in the performance of state-of-the-art attacks. For instance, these techniques can merely achieve 20\% in the physical world and 30.8\% in the digital world while our attack provides superior success rate of up to 65\% and 84.56\%, respectively when attacking the YOLOv3tiny detector deployed in smart cameras at the edge.



## **20. Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation**

cs.CL

14 pages, 4 figures

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2305.11596v1) [paper-pdf](http://arxiv.org/pdf/2305.11596v1)

**Authors**: Xuanli He, Qiongkai Xu, Jun Wang, Benjamin Rubinstein, Trevor Cohn

**Abstract**: Modern NLP models are often trained over large untrusted datasets, raising the potential for a malicious adversary to compromise model behaviour. For instance, backdoors can be implanted through crafting training instances with a specific textual trigger and a target label. This paper posits that backdoor poisoning attacks exhibit spurious correlation between simple text features and classification labels, and accordingly, proposes methods for mitigating spurious correlation as means of defence. Our empirical study reveals that the malicious triggers are highly correlated to their target labels; therefore such correlations are extremely distinguishable compared to those scores of benign features, and can be used to filter out potentially problematic instances. Compared with several existing defences, our defence method significantly reduces attack success rates across backdoor attacks, and in the case of insertion based attacks, our method provides a near-perfect defence.



## **21. Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning**

cs.LG

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2304.10783v2) [paper-pdf](http://arxiv.org/pdf/2304.10783v2)

**Authors**: Hangtao Zhang, Zeming Yao, Leo Yu Zhang, Shengshan Hu, Chao Chen, Alan Liew, Zhetao Li

**Abstract**: Federated learning (FL) is vulnerable to poisoning attacks, where adversaries corrupt the global aggregation results and cause denial-of-service (DoS). Unlike recent model poisoning attacks that optimize the amplitude of malicious perturbations along certain prescribed directions to cause DoS, we propose a Flexible Model Poisoning Attack (FMPA) that can achieve versatile attack goals. We consider a practical threat scenario where no extra knowledge about the FL system (e.g., aggregation rules or updates on benign devices) is available to adversaries. FMPA exploits the global historical information to construct an estimator that predicts the next round of the global model as a benign reference. It then fine-tunes the reference model to obtain the desired poisoned model with low accuracy and small perturbations. Besides the goal of causing DoS, FMPA can be naturally extended to launch a fine-grained controllable attack, making it possible to precisely reduce the global accuracy. Armed with precise control, malicious FL service providers can gain advantages over their competitors without getting noticed, hence opening a new attack surface in FL other than DoS. Even for the purpose of DoS, experiments show that FMPA significantly decreases the global accuracy, outperforming six state-of-the-art attacks.The code can be found at https://github.com/ZhangHangTao/Poisoning-Attack-on-FL.



## **22. Free Lunch for Privacy Preserving Distributed Graph Learning**

cs.LG

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2305.10869v2) [paper-pdf](http://arxiv.org/pdf/2305.10869v2)

**Authors**: Nimesh Agrawal, Nikita Malik, Sandeep Kumar

**Abstract**: Learning on graphs is becoming prevalent in a wide range of applications including social networks, robotics, communication, medicine, etc. These datasets belonging to entities often contain critical private information. The utilization of data for graph learning applications is hampered by the growing privacy concerns from users on data sharing. Existing privacy-preserving methods pre-process the data to extract user-side features, and only these features are used for subsequent learning. Unfortunately, these methods are vulnerable to adversarial attacks to infer private attributes. We present a novel privacy-respecting framework for distributed graph learning and graph-based machine learning. In order to perform graph learning and other downstream tasks on the server side, this framework aims to learn features as well as distances without requiring actual features while preserving the original structural properties of the raw data. The proposed framework is quite generic and highly adaptable. We demonstrate the utility of the Euclidean space, but it can be applied with any existing method of distance approximation and graph learning for the relevant spaces. Through extensive experimentation on both synthetic and real datasets, we demonstrate the efficacy of the framework in terms of comparing the results obtained without data sharing to those obtained with data sharing as a benchmark. This is, to our knowledge, the first privacy-preserving distributed graph learning framework.



## **23. Security of Nakamoto Consensus under Congestion**

cs.CR

**SubmitDate**: 2023-05-19    [abs](http://arxiv.org/abs/2303.09113v2) [paper-pdf](http://arxiv.org/pdf/2303.09113v2)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: Nakamoto consensus (NC) powers major proof-of-work (PoW) and proof-of-stake (PoS) blockchains such as Bitcoin or Cardano. Given a network of nodes with certain communication and computation capacities, against what fraction of adversarial power (the resilience) is Nakamoto consensus secure for a given block production rate? Prior security analyses of NC used a bounded delay model which does not capture network congestion resulting from high block production rates, bursty release of adversarial blocks, and in PoS, spamming due to equivocations. For PoW, we find a new attack, called teasing attack, that exploits congestion to increase the time taken to download and verify blocks, thereby succeeding at lower adversarial power than the private attack which was deemed to be the worst-case attack in prior analysis. By adopting a bounded bandwidth model to capture congestion, and through an improved analysis method, we identify the resilience of PoW NC for a given block production rate. In PoS, we augment our attack with equivocations to further increase congestion, making the vanilla PoS NC protocol insecure against any adversarial power except at very low block production rates. To counter equivocation spamming in PoS, we present a new NC-style protocol Sanitizing PoS (SaPoS) which achieves the same resilience as PoW NC.



## **24. Quantifying the robustness of deep multispectral segmentation models against natural perturbations and data poisoning**

cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11347v1) [paper-pdf](http://arxiv.org/pdf/2305.11347v1)

**Authors**: Elise Bishoff, Charles Godfrey, Myles McKay, Eleanor Byler

**Abstract**: In overhead image segmentation tasks, including additional spectral bands beyond the traditional RGB channels can improve model performance. However, it is still unclear how incorporating this additional data impacts model robustness to adversarial attacks and natural perturbations. For adversarial robustness, the additional information could improve the model's ability to distinguish malicious inputs, or simply provide new attack avenues and vulnerabilities. For natural perturbations, the additional information could better inform model decisions and weaken perturbation effects or have no significant influence at all. In this work, we seek to characterize the performance and robustness of a multispectral (RGB and near infrared) image segmentation model subjected to adversarial attacks and natural perturbations. While existing adversarial and natural robustness research has focused primarily on digital perturbations, we prioritize on creating realistic perturbations designed with physical world conditions in mind. For adversarial robustness, we focus on data poisoning attacks whereas for natural robustness, we focus on extending ImageNet-C common corruptions for fog and snow that coherently and self-consistently perturbs the input data. Overall, we find both RGB and multispectral models are vulnerable to data poisoning attacks regardless of input or fusion architectures and that while physically realizable natural perturbations still degrade model performance, the impact differs based on fusion architecture and input data.



## **25. On the Noise Stability and Robustness of Adversarially Trained Networks on NVM Crossbars**

cs.LG

13 pages, 14 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2109.09060v2) [paper-pdf](http://arxiv.org/pdf/2109.09060v2)

**Authors**: Chun Tao, Deboleena Roy, Indranil Chakraborty, Kaushik Roy

**Abstract**: Applications based on Deep Neural Networks (DNNs) have grown exponentially in the past decade. To match their increasing computational needs, several Non-Volatile Memory (NVM) crossbar based accelerators have been proposed. Recently, researchers have shown that apart from improved energy efficiency and performance, such approximate hardware also possess intrinsic robustness for defense against adversarial attacks. Prior works quantified this intrinsic robustness for vanilla DNNs trained on unperturbed inputs. However, adversarial training of DNNs is the benchmark technique for robustness, and sole reliance on intrinsic robustness of the hardware may not be sufficient. In this work, we explore the design of robust DNNs through the amalgamation of adversarial training and intrinsic robustness of NVM crossbar-based analog hardware. First, we study the noise stability of such networks on unperturbed inputs and observe that internal activations of adversarially trained networks have lower Signal-to-Noise Ratio (SNR), and are sensitive to noise compared to vanilla networks. As a result, they suffer on average 2x performance degradation due to the approximate computations on analog hardware. Noise stability analyses show the instability of adversarially trained DNNs. On the other hand, for adversarial images generated using Square Black Box attacks, ResNet-10/20 adversarially trained on CIFAR-10/100 display a robustness gain of 20-30%. For adversarial images generated using Projected-Gradient-Descent (PGD) White-Box attacks, adversarially trained DNNs present a 5-10% gain in robust accuracy due to underlying NVM crossbar when $\epsilon_{attack}$ is greater than $\epsilon_{train}$. Our results indicate that implementing adversarially trained networks on analog hardware requires careful calibration between hardware non-idealities and $\epsilon_{train}$ for optimum robustness and performance.



## **26. TrustSER: On the Trustworthiness of Fine-tuning Pre-trained Speech Embeddings For Speech Emotion Recognition**

cs.SD

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11229v1) [paper-pdf](http://arxiv.org/pdf/2305.11229v1)

**Authors**: Tiantian Feng, Rajat Hebbar, Shrikanth Narayanan

**Abstract**: Recent studies have explored the use of pre-trained embeddings for speech emotion recognition (SER), achieving comparable performance to conventional methods that rely on low-level knowledge-inspired acoustic features. These embeddings are often generated from models trained on large-scale speech datasets using self-supervised or weakly-supervised learning objectives. Despite the significant advancements made in SER through the use of pre-trained embeddings, there is a limited understanding of the trustworthiness of these methods, including privacy breaches, unfair performance, vulnerability to adversarial attacks, and computational cost, all of which may hinder the real-world deployment of these systems. In response, we introduce TrustSER, a general framework designed to evaluate the trustworthiness of SER systems using deep learning methods, with a focus on privacy, safety, fairness, and sustainability, offering unique insights into future research in the field of SER. Our code is publicly available under: https://github.com/usc-sail/trust-ser.



## **27. Attacks on Online Learners: a Teacher-Student Analysis**

stat.ML

15 pages, 6 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11132v1) [paper-pdf](http://arxiv.org/pdf/2305.11132v1)

**Authors**: Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti

**Abstract**: Machine learning models are famously vulnerable to adversarial attacks: small ad-hoc perturbations of the data that can catastrophically alter the model predictions. While a large literature has studied the case of test-time attacks on pre-trained models, the important case of attacks in an online learning setting has received little attention so far. In this work, we use a control-theoretical perspective to study the scenario where an attacker may perturb data labels to manipulate the learning dynamics of an online learner. We perform a theoretical analysis of the problem in a teacher-student setup, considering different attack strategies, and obtaining analytical results for the steady state of simple linear learners. These results enable us to prove that a discontinuous transition in the learner's accuracy occurs when the attack strength exceeds a critical threshold. We then study empirically attacks on learners with complex architectures using real data, confirming the insights of our theoretical analysis. Our findings show that greedy attacks can be extremely efficient, especially when data stream in small batches.



## **28. Deep PackGen: A Deep Reinforcement Learning Framework for Adversarial Network Packet Generation**

cs.CR

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11039v1) [paper-pdf](http://arxiv.org/pdf/2305.11039v1)

**Authors**: Soumyadeep Hore, Jalal Ghadermazi, Diwas Paudel, Ankit Shah, Tapas K. Das, Nathaniel D. Bastian

**Abstract**: Recent advancements in artificial intelligence (AI) and machine learning (ML) algorithms, coupled with the availability of faster computing infrastructure, have enhanced the security posture of cybersecurity operations centers (defenders) through the development of ML-aided network intrusion detection systems (NIDS). Concurrently, the abilities of adversaries to evade security have also increased with the support of AI/ML models. Therefore, defenders need to proactively prepare for evasion attacks that exploit the detection mechanisms of NIDS. Recent studies have found that the perturbation of flow-based and packet-based features can deceive ML models, but these approaches have limitations. Perturbations made to the flow-based features are difficult to reverse-engineer, while samples generated with perturbations to the packet-based features are not playable.   Our methodological framework, Deep PackGen, employs deep reinforcement learning to generate adversarial packets and aims to overcome the limitations of approaches in the literature. By taking raw malicious network packets as inputs and systematically making perturbations on them, Deep PackGen camouflages them as benign packets while still maintaining their functionality. In our experiments, using publicly available data, Deep PackGen achieved an average adversarial success rate of 66.4\% against various ML models and across different attack types. Our investigation also revealed that more than 45\% of the successful adversarial samples were out-of-distribution packets that evaded the decision boundaries of the classifiers. The knowledge gained from our study on the adversary's ability to make specific evasive perturbations to different types of malicious packets can help defenders enhance the robustness of their NIDS against evolving adversarial attacks.



## **29. SoK: Data Privacy in Virtual Reality**

cs.HC

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2301.05940v2) [paper-pdf](http://arxiv.org/pdf/2301.05940v2)

**Authors**: Gonzalo Munilla Garrido, Vivek Nair, Dawn Song

**Abstract**: The adoption of virtual reality (VR) technologies has rapidly gained momentum in recent years as companies around the world begin to position the so-called "metaverse" as the next major medium for accessing and interacting with the internet. While consumers have become accustomed to a degree of data harvesting on the web, the real-time nature of data sharing in the metaverse indicates that privacy concerns are likely to be even more prevalent in the new "Web 3.0." Research into VR privacy has demonstrated that a plethora of sensitive personal information is observable by various would-be adversaries from just a few minutes of telemetry data. On the other hand, we have yet to see VR parallels for many privacy-preserving tools aimed at mitigating threats on conventional platforms. This paper aims to systematize knowledge on the landscape of VR privacy threats and countermeasures by proposing a comprehensive taxonomy of data attributes, protections, and adversaries based on the study of 68 collected publications. We complement our qualitative discussion with a statistical analysis of the risk associated with various data sources inherent to VR in consideration of the known attacks and defenses. By focusing on highlighting the clear outstanding opportunities, we hope to motivate and guide further research into this increasingly important field.



## **30. Certified Robust Neural Networks: Generalization and Corruption Resistance**

stat.ML

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2303.02251v2) [paper-pdf](http://arxiv.org/pdf/2303.02251v2)

**Authors**: Amine Bennouna, Ryan Lucas, Bart Van Parys

**Abstract**: Recent work have demonstrated that robustness (to "corruption") can be at odds with generalization. Adversarial training, for instance, aims to reduce the problematic susceptibility of modern neural networks to small data perturbations. Surprisingly, overfitting is a major concern in adversarial training despite being mostly absent in standard training. We provide here theoretical evidence for this peculiar "robust overfitting" phenomenon. Subsequently, we advance a novel distributionally robust loss function bridging robustness and generalization. We demonstrate both theoretically as well as empirically the loss to enjoy a certified level of robustness against two common types of corruption--data evasion and poisoning attacks--while ensuring guaranteed generalization. We show through careful numerical experiments that our resulting holistic robust (HR) training procedure yields SOTA performance. Finally, we indicate that HR training can be interpreted as a direct extension of adversarial training and comes with a negligible additional computational burden. A ready-to-use python library implementing our algorithm is available at https://github.com/RyanLucas3/HR_Neural_Networks.



## **31. Architecture-agnostic Iterative Black-box Certified Defense against Adversarial Patches**

cs.CV

9 pages

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10929v1) [paper-pdf](http://arxiv.org/pdf/2305.10929v1)

**Authors**: Di Yang, Yihao Huang, Qing Guo, Felix Juefei-Xu, Ming Hu, Yang Liu, Geguang Pu

**Abstract**: The adversarial patch attack aims to fool image classifiers within a bounded, contiguous region of arbitrary changes, posing a real threat to computer vision systems (e.g., autonomous driving, content moderation, biometric authentication, medical imaging) in the physical world. To address this problem in a trustworthy way, proposals have been made for certified patch defenses that ensure the robustness of classification models and prevent future patch attacks from breaching the defense. State-of-the-art certified defenses can be compatible with any model architecture, as well as achieve high clean and certified accuracy. Although the methods are adaptive to arbitrary patch positions, they inevitably need to access the size of the adversarial patch, which is unreasonable and impractical in real-world attack scenarios. To improve the feasibility of the architecture-agnostic certified defense in a black-box setting (i.e., position and size of the patch are both unknown), we propose a novel two-stage Iterative Black-box Certified Defense method, termed IBCD.In the first stage, it estimates the patch size in a search-based manner by evaluating the size relationship between the patch and mask with pixel masking. In the second stage, the accuracy results are calculated by the existing white-box certified defense methods with the estimated patch size. The experiments conducted on two popular model architectures and two datasets verify the effectiveness and efficiency of IBCD.



## **32. How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses**

cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10862v1) [paper-pdf](http://arxiv.org/pdf/2305.10862v1)

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Deep Learning is currently used to perform multiple tasks, such as object recognition, face recognition, and natural language processing. However, Deep Neural Networks (DNNs) are vulnerable to perturbations that alter the network prediction (adversarial examples), raising concerns regarding its usage in critical areas, such as self-driving vehicles, malware detection, and healthcare. This paper compiles the most recent adversarial attacks, grouped by the attacker capacity, and modern defenses clustered by protection strategies. We also present the new advances regarding Vision Transformers, summarize the datasets and metrics used in the context of adversarial settings, and compare the state-of-the-art results under different attacks, finishing with the identification of open issues.



## **33. Towards an Accurate and Secure Detector against Adversarial Perturbations**

cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10856v1) [paper-pdf](http://arxiv.org/pdf/2305.10856v1)

**Authors**: Chao Wang, Shuren Qi, Zhiqiu Huang, Yushu Zhang, Xiaochun Cao

**Abstract**: The vulnerability of deep neural networks to adversarial perturbations has been widely perceived in the computer vision community. From a security perspective, it poses a critical risk for modern vision systems, e.g., the popular Deep Learning as a Service (DLaaS) frameworks. For protecting off-the-shelf deep models while not modifying them, current algorithms typically detect adversarial patterns through discriminative decomposition of natural-artificial data. However, these decompositions are biased towards frequency or spatial discriminability, thus failing to capture subtle adversarial patterns comprehensively. More seriously, they are typically invertible, meaning successful defense-aware (secondary) adversarial attack (i.e., evading the detector as well as fooling the model) is practical under the assumption that the adversary is fully aware of the detector (i.e., the Kerckhoffs's principle). Motivated by such facts, we propose an accurate and secure adversarial example detector, relying on a spatial-frequency discriminative decomposition with secret keys. It expands the above works on two aspects: 1) the introduced Krawtchouk basis provides better spatial-frequency discriminability and thereby is more suitable for capturing adversarial patterns than the common trigonometric or wavelet basis; 2) the extensive parameters for decomposition are generated by a pseudo-random function with secret keys, hence blocking the defense-aware adversarial attack. Theoretical and numerical analysis demonstrates the increased accuracy and security of our detector w.r.t. a number of state-of-the-art algorithms.



## **34. Adversarial Scratches: Deployable Attacks to CNN Classifiers**

cs.LG

This work is published at Pattern Recognition (Elsevier). This paper  stems from 'Scratch that! An Evolution-based Adversarial Attack against  Neural Networks' for which an arXiv preprint is available at  arXiv:1912.02316. Further studies led to a complete overhaul of the work,  resulting in this paper

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2204.09397v3) [paper-pdf](http://arxiv.org/pdf/2204.09397v3)

**Authors**: Loris Giulivi, Malhar Jere, Loris Rossi, Farinaz Koushanfar, Gabriela Ciocarlie, Briland Hitaj, Giacomo Boracchi

**Abstract**: A growing body of work has shown that deep neural networks are susceptible to adversarial examples. These take the form of small perturbations applied to the model's input which lead to incorrect predictions. Unfortunately, most literature focuses on visually imperceivable perturbations to be applied to digital images that often are, by design, impossible to be deployed to physical targets. We present Adversarial Scratches: a novel L0 black-box attack, which takes the form of scratches in images, and which possesses much greater deployability than other state-of-the-art attacks. Adversarial Scratches leverage B\'ezier Curves to reduce the dimension of the search space and possibly constrain the attack to a specific location. We test Adversarial Scratches in several scenarios, including a publicly available API and images of traffic signs. Results show that, often, our attack achieves higher fooling rate than other deployable state-of-the-art methods, while requiring significantly fewer queries and modifying very few pixels.



## **35. Adversarial Amendment is the Only Force Capable of Transforming an Enemy into a Friend**

cs.AI

Accepted to IJCAI 2023, 10 pages, 5 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10766v1) [paper-pdf](http://arxiv.org/pdf/2305.10766v1)

**Authors**: Chong Yu, Tao Chen, Zhongxue Gan

**Abstract**: Adversarial attack is commonly regarded as a huge threat to neural networks because of misleading behavior. This paper presents an opposite perspective: adversarial attacks can be harnessed to improve neural models if amended correctly. Unlike traditional adversarial defense or adversarial training schemes that aim to improve the adversarial robustness, the proposed adversarial amendment (AdvAmd) method aims to improve the original accuracy level of neural models on benign samples. We thoroughly analyze the distribution mismatch between the benign and adversarial samples. This distribution mismatch and the mutual learning mechanism with the same learning ratio applied in prior art defense strategies is the main cause leading the accuracy degradation for benign samples. The proposed AdvAmd is demonstrated to steadily heal the accuracy degradation and even leads to a certain accuracy boost of common neural models on benign classification, object detection, and segmentation tasks. The efficacy of the AdvAmd is contributed by three key components: mediate samples (to reduce the influence of distribution mismatch with a fine-grained amendment), auxiliary batch norm (to solve the mutual learning mechanism and the smoother judgment surface), and AdvAmd loss (to adjust the learning ratios according to different attack vulnerabilities) through quantitative and ablation experiments.



## **36. Re-thinking Data Availablity Attacks Against Deep Neural Networks**

cs.CR

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10691v1) [paper-pdf](http://arxiv.org/pdf/2305.10691v1)

**Authors**: Bin Fang, Bo Li, Shuang Wu, Ran Yi, Shouhong Ding, Lizhuang Ma

**Abstract**: The unauthorized use of personal data for commercial purposes and the clandestine acquisition of private data for training machine learning models continue to raise concerns. In response to these issues, researchers have proposed availability attacks that aim to render data unexploitable. However, many current attack methods are rendered ineffective by adversarial training. In this paper, we re-examine the concept of unlearnable examples and discern that the existing robust error-minimizing noise presents an inaccurate optimization objective. Building on these observations, we introduce a novel optimization paradigm that yields improved protection results with reduced computational time requirements. We have conducted extensive experiments to substantiate the soundness of our approach. Moreover, our method establishes a robust foundation for future research in this area.



## **37. Content-based Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10665v1) [paper-pdf](http://arxiv.org/pdf/2305.10665v1)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4% and 16.8-48.0% in normally trained models and defense methods, respectively.



## **38. Exact Recovery for System Identification with More Corrupt Data than Clean Data**

cs.LG

24 pages, 2 figures

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2305.10506v1) [paper-pdf](http://arxiv.org/pdf/2305.10506v1)

**Authors**: Baturalp Yalcin, Javad Lavaei, Murat Arcak

**Abstract**: In this paper, we study the system identification problem for linear discrete-time systems under adversaries and analyze two lasso-type estimators. We study both asymptotic and non-asymptotic properties of these estimators in two separate scenarios, corresponding to deterministic and stochastic models for the attack times. Since the samples collected from the system are correlated, the existing results on lasso are not applicable. We show that when the system is stable and the attacks are injected periodically, the sample complexity for the exact recovery of the system dynamics is O(n), where n is the dimension of the states. When the adversarial attacks occur at each time instance with probability p, the required sample complexity for the exact recovery scales as O(\log(n)p/(1-p)^2). This result implies the almost sure convergence to the true system dynamics under the asymptotic regime. As a by-product, even when more than half of the data is compromised, our estimators still learn the system correctly. This paper provides the first mathematical guarantee in the literature on learning from correlated data for dynamical systems in the case when there is less clean data than corrupt data.



## **39. Raising the Bar for Certified Adversarial Robustness with Diffusion Models**

cs.LG

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2305.10388v1) [paper-pdf](http://arxiv.org/pdf/2305.10388v1)

**Authors**: Thomas Altstidl, David Dobre, Björn Eskofier, Gauthier Gidel, Leo Schwinn

**Abstract**: Certified defenses against adversarial attacks offer formal guarantees on the robustness of a model, making them more reliable than empirical methods such as adversarial training, whose effectiveness is often later reduced by unseen attacks. Still, the limited certified robustness that is currently achievable has been a bottleneck for their practical adoption. Gowal et al. and Wang et al. have shown that generating additional training data using state-of-the-art diffusion models can considerably improve the robustness of adversarial training. In this work, we demonstrate that a similar approach can substantially improve deterministic certified defenses. In addition, we provide a list of recommendations to scale the robustness of certified training approaches. One of our main insights is that the generalization gap, i.e., the difference between the training and test accuracy of the original model, is a good predictor of the magnitude of the robustness improvement when using additional generated data. Our approach achieves state-of-the-art deterministic robustness certificates on CIFAR-10 for the $\ell_2$ ($\epsilon = 36/255$) and $\ell_\infty$ ($\epsilon = 8/255$) threat models, outperforming the previous best results by $+3.95\%$ and $+1.39\%$, respectively. Furthermore, we report similar improvements for CIFAR-100.



## **40. Certified Invertibility in Neural Networks via Mixed-Integer Programming**

cs.LG

22 pages, 7 figures

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2301.11783v2) [paper-pdf](http://arxiv.org/pdf/2301.11783v2)

**Authors**: Tianqi Cui, Thomas Bertalan, George J. Pappas, Manfred Morari, Ioannis G. Kevrekidis, Mahyar Fazlyab

**Abstract**: Neural networks are known to be vulnerable to adversarial attacks, which are small, imperceptible perturbations that can significantly alter the network's output. Conversely, there may exist large, meaningful perturbations that do not affect the network's decision (excessive invariance). In our research, we investigate this latter phenomenon in two contexts: (a) discrete-time dynamical system identification, and (b) the calibration of a neural network's output to that of another network. We examine noninvertibility through the lens of mathematical optimization, where the global solution measures the ``safety" of the network predictions by their distance from the non-invertibility boundary. We formulate mixed-integer programs (MIPs) for ReLU networks and $L_p$ norms ($p=1,2,\infty$) that apply to neural network approximators of dynamical systems. We also discuss how our findings can be useful for invertibility certification in transformations between neural networks, e.g. between different levels of network pruning.



## **41. Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures**

cs.IR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.08183v2) [paper-pdf](http://arxiv.org/pdf/2305.08183v2)

**Authors**: Wei Yuan, Shilong Yuan, Chaoqun Yang, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated recommender systems (FedRecs) have been widely explored recently due to their ability to protect user data privacy. In FedRecs, a central server collaboratively learns recommendation models by sharing model public parameters with clients, thereby offering a privacy-preserving solution. Unfortunately, the exposure of model parameters leaves a backdoor for adversaries to manipulate FedRecs. Existing works about FedRec security already reveal that items can easily be promoted by malicious users via model poisoning attacks, but all of them mainly focus on FedRecs with only collaborative information (i.e., user-item interactions). We argue that these attacks are effective because of the data sparsity of collaborative signals. In practice, auxiliary information, such as products' visual descriptions, is used to alleviate collaborative filtering data's sparsity. Therefore, when incorporating visual information in FedRecs, all existing model poisoning attacks' effectiveness becomes questionable. In this paper, we conduct extensive experiments to verify that incorporating visual information can beat existing state-of-the-art attacks in reasonable settings. However, since visual information is usually provided by external sources, simply including it will create new security problems. Specifically, we propose a new kind of poisoning attack for visually-aware FedRecs, namely image poisoning attacks, where adversaries can gradually modify the uploaded image to manipulate item ranks during FedRecs' training process. Furthermore, we reveal that the potential collaboration between image poisoning attacks and model poisoning attacks will make visually-aware FedRecs more vulnerable to being manipulated. To safely use visual information, we employ a diffusion model in visually-aware FedRecs to purify each uploaded image and detect the adversarial images.



## **42. A theoretical basis for Blockchain Extractable Value**

cs.CR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02154v3) [paper-pdf](http://arxiv.org/pdf/2302.02154v3)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Extractable Value refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream protocols, like e.g. decentralized exchanges, are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the growing impact of these attacks in the real world, theoretical foundations are still missing. We propose a formal theory of Extractable Value, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against Extractable Value attacks.



## **43. Exploring the Connection between Robust and Generative Models**

cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2304.04033v3) [paper-pdf](http://arxiv.org/pdf/2304.04033v3)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.



## **44. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

cs.LG

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09241v1) [paper-pdf](http://arxiv.org/pdf/2305.09241v1)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning.



## **45. Iterative Adversarial Attack on Image-guided Story Ending Generation**

cs.CV

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.13208v1) [paper-pdf](http://arxiv.org/pdf/2305.13208v1)

**Authors**: Youze Wang, Wenbo Hu, Richang Hong

**Abstract**: Multimodal learning involves developing models that can integrate information from various sources like images and texts. In this field, multimodal text generation is a crucial aspect that involves processing data from multiple modalities and outputting text. The image-guided story ending generation (IgSEG) is a particularly significant task, targeting on an understanding of complex relationships between text and image data with a complete story text ending. Unfortunately, deep neural networks, which are the backbone of recent IgSEG models, are vulnerable to adversarial samples. Current adversarial attack methods mainly focus on single-modality data and do not analyze adversarial attacks for multimodal text generation tasks that use cross-modal information. To this end, we propose an iterative adversarial attack method (Iterative-attack) that fuses image and text modality attacks, allowing for an attack search for adversarial text and image in an more effective iterative way. Experimental results demonstrate that the proposed method outperforms existing single-modal and non-iterative multimodal attack methods, indicating the potential for improving the adversarial robustness of multimodal text generation models, such as multimodal machine translation, multimodal question answering, etc.



## **46. Ortho-ODE: Enhancing Robustness and of Neural ODEs against Adversarial Attacks**

cs.LG

Final project paper

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09179v1) [paper-pdf](http://arxiv.org/pdf/2305.09179v1)

**Authors**: Vishal Purohit

**Abstract**: Neural Ordinary Differential Equations (NODEs) probed the usage of numerical solvers to solve the differential equation characterized by a Neural Network (NN), therefore initiating a new paradigm of deep learning models with infinite depth. NODEs were designed to tackle the irregular time series problem. However, NODEs have demonstrated robustness against various noises and adversarial attacks. This paper is about the natural robustness of NODEs and examines the cause behind such surprising behaviour. We show that by controlling the Lipschitz constant of the ODE dynamics the robustness can be significantly improved. We derive our approach from Grownwall's inequality. Further, we draw parallels between contractivity theory and Grownwall's inequality. Experimentally we corroborate the enhanced robustness on numerous datasets - MNIST, CIFAR-10, and CIFAR 100. We also present the impact of adaptive and non-adaptive solvers on the robustness of NODEs.



## **47. Run-Off Election: Improved Provable Defense against Data Poisoning Attacks**

cs.LG

Accepted to ICML 2023

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02300v3) [paper-pdf](http://arxiv.org/pdf/2302.02300v3)

**Authors**: Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi

**Abstract**: In data poisoning attacks, an adversary tries to change a model's prediction by adding, modifying, or removing samples in the training data. Recently, ensemble-based approaches for obtaining provable defenses against data poisoning have been proposed where predictions are done by taking a majority vote across multiple base models. In this work, we show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. Based on this approach, we propose DPA+ROE and FA+ROE defense methods based on Deep Partition Aggregation (DPA) and Finite Aggregation (FA) approaches from prior work. We evaluate our methods on MNIST, CIFAR-10, and GTSRB and obtain improvements in certified accuracy by up to 3%-4%. Also, by applying ROE on a boosted version of DPA, we gain improvements around 12%-27% comparing to the current state-of-the-art, establishing a new state-of-the-art in (pointwise) certified robustness against data poisoning. In many cases, our approach outperforms the state-of-the-art, even when using 32 times less computational power.



## **48. Training Neural Networks without Backpropagation: A Deeper Dive into the Likelihood Ratio Method**

cs.LG

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08960v1) [paper-pdf](http://arxiv.org/pdf/2305.08960v1)

**Authors**: Jinyang Jiang, Zeliang Zhang, Chenliang Xu, Zhaofei Yu, Yijie Peng

**Abstract**: Backpropagation (BP) is the most important gradient estimation method for training neural networks in deep learning. However, the literature shows that neural networks trained by BP are vulnerable to adversarial attacks. We develop the likelihood ratio (LR) method, a new gradient estimation method, for training a broad range of neural network architectures, including convolutional neural networks, recurrent neural networks, graph neural networks, and spiking neural networks, without recursive gradient computation. We propose three methods to efficiently reduce the variance of the gradient estimation in the neural network training process. Our experiments yield numerical results for training different neural networks on several datasets. All results demonstrate that the LR method is effective for training various neural networks and significantly improves the robustness of the neural networks under adversarial attacks relative to the BP method.



## **49. Attacking Perceptual Similarity Metrics**

cs.CV

TMLR 2023 (Featured Certification). Code is available at  https://tinyurl.com/attackingpsm

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08840v1) [paper-pdf](http://arxiv.org/pdf/2305.08840v1)

**Authors**: Abhijay Ghildyal, Feng Liu

**Abstract**: Perceptual similarity metrics have progressively become more correlated with human judgments on perceptual similarity; however, despite recent advances, the addition of an imperceptible distortion can still compromise these metrics. In our study, we systematically examine the robustness of these metrics to imperceptible adversarial perturbations. Following the two-alternative forced-choice experimental design with two distorted images and one reference image, we perturb the distorted image closer to the reference via an adversarial attack until the metric flips its judgment. We first show that all metrics in our study are susceptible to perturbations generated via common adversarial attacks such as FGSM, PGD, and the One-pixel attack. Next, we attack the widely adopted LPIPS metric using spatial-transformation-based adversarial perturbations (stAdv) in a white-box setting to craft adversarial examples that can effectively transfer to other similarity metrics in a black-box setting. We also combine the spatial attack stAdv with PGD ($\ell_\infty$-bounded) attack to increase transferability and use these adversarial examples to benchmark the robustness of both traditional and recently developed metrics. Our benchmark provides a good starting point for discussion and further research on the robustness of metrics to imperceptible adversarial perturbations.



## **50. Defending Against Misinformation Attacks in Open-Domain Question Answering**

cs.CL

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2212.10002v2) [paper-pdf](http://arxiv.org/pdf/2212.10002v2)

**Authors**: Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, Benjamin Van Durme

**Abstract**: Recent work in open-domain question answering (ODQA) has shown that adversarial poisoning of the search collection can cause large drops in accuracy for production systems. However, little to no work has proposed methods to defend against these attacks. To do so, we rely on the intuition that redundant information often exists in large corpora. To find it, we introduce a method that uses query augmentation to search for a diverse set of passages that could answer the original question but are less likely to have been poisoned. We integrate these new passages into the model through the design of a novel confidence method, comparing the predicted answer to its appearance in the retrieved contexts (what we call \textit{Confidence from Answer Redundancy}, i.e. CAR). Together these methods allow for a simple but effective way to defend against poisoning attacks that provides gains of nearly 20\% exact match across varying levels of data poisoning/knowledge conflicts.



