# Latest Adversarial Attack Papers
**update at 2023-12-18 10:04:44**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models**

cs.LG

19 pages, 6 figures; AAAI24

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2305.12066v2) [paper-pdf](http://arxiv.org/pdf/2305.12066v2)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-task learning (MTL) creates a single machine learning model called multi-task model to simultaneously perform multiple tasks. Although the security of single task classifiers has been extensively studied, there are several critical security research questions for multi-task models including 1) How secure are multi-task models to single task adversarial machine learning attacks, 2) Can adversarial attacks be designed to attack multiple tasks simultaneously, and 3) Does task sharing and adversarial training increase multi-task model robustness to adversarial attacks? In this paper, we answer these questions through careful analysis and rigorous experimentation. First, we develop na\"ive adaptation of single-task white-box attacks and analyze their inherent drawbacks. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking a multi-task model as an optimization problem based on averaged relative loss change, which can be solved by approximating the problem as an integer linear programming problem. Extensive evaluation on two popular MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrates the effectiveness of DGBA compared to na\"ive multi-task attack baselines on both clean and adversarially trained multi-task models. The results also reveal a fundamental trade-off between improving task accuracy by sharing parameters across tasks and undermining model robustness due to increased attack transferability from parameter sharing. DGBA is open-sourced and available at https://github.com/zhanglijun95/MTLAttack-DGBA.



## **2. Exploring Adversarial Robustness of Vision Transformers in the Spectral Perspective**

cs.CV

Accepted in IEEE/CVF Winter Conference on Applications of Computer  Vision (WACV) 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2208.09602v2) [paper-pdf](http://arxiv.org/pdf/2208.09602v2)

**Authors**: Gihyun Kim, Juyeop Kim, Jong-Seok Lee

**Abstract**: The Vision Transformer has emerged as a powerful tool for image classification tasks, surpassing the performance of convolutional neural networks (CNNs). Recently, many researchers have attempted to understand the robustness of Transformers against adversarial attacks. However, previous researches have focused solely on perturbations in the spatial domain. This paper proposes an additional perspective that explores the adversarial robustness of Transformers against frequency-selective perturbations in the spectral domain. To facilitate comparison between these two domains, an attack framework is formulated as a flexible tool for implementing attacks on images in the spatial and spectral domains. The experiments reveal that Transformers rely more on phase and low frequency information, which can render them more vulnerable to frequency-selective attacks than CNNs. This work offers new insights into the properties and adversarial robustness of Transformers.



## **3. LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer**

cs.CV

13 pages, 3 figures. Accepted to AAAI 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09935v1) [paper-pdf](http://arxiv.org/pdf/2312.09935v1)

**Authors**: Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu

**Abstract**: Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.



## **4. A Game-theoretic Framework for Privacy-preserving Federated Learning**

cs.LG

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2304.05836v2) [paper-pdf](http://arxiv.org/pdf/2304.05836v2)

**Authors**: Xiaojin Zhang, Lixin Fan, Siwei Wang, Wenjie Li, Kai Chen, Qiang Yang

**Abstract**: In federated learning, benign participants aim to optimize a global model collaboratively. However, the risk of \textit{privacy leakage} cannot be ignored in the presence of \textit{semi-honest} adversaries. Existing research has focused either on designing protection mechanisms or on inventing attacking mechanisms. While the battle between defenders and attackers seems never-ending, we are concerned with one critical question: is it possible to prevent potential attacks in advance? To address this, we propose the first game-theoretic framework that considers both FL defenders and attackers in terms of their respective payoffs, which include computational costs, FL model utilities, and privacy leakage risks. We name this game the federated learning privacy game (FLPG), in which neither defenders nor attackers are aware of all participants' payoffs.   To handle the \textit{incomplete information} inherent in this situation, we propose associating the FLPG with an \textit{oracle} that has two primary responsibilities. First, the oracle provides lower and upper bounds of the payoffs for the players. Second, the oracle acts as a correlation device, privately providing suggested actions to each player. With this novel framework, we analyze the optimal strategies of defenders and attackers. Furthermore, we derive and demonstrate conditions under which the attacker, as a rational decision-maker, should always follow the oracle's suggestion \textit{not to attack}.



## **5. FlowMur: A Stealthy and Practical Audio Backdoor Attack with Limited Knowledge**

cs.CR

To appear at lEEE Symposium on Security & Privacy (Oakland) 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09665v1) [paper-pdf](http://arxiv.org/pdf/2312.09665v1)

**Authors**: Jiahe Lan, Jie Wang, Baochen Yan, Zheng Yan, Elisa Bertino

**Abstract**: Speech recognition systems driven by DNNs have revolutionized human-computer interaction through voice interfaces, which significantly facilitate our daily lives. However, the growing popularity of these systems also raises special concerns on their security, particularly regarding backdoor attacks. A backdoor attack inserts one or more hidden backdoors into a DNN model during its training process, such that it does not affect the model's performance on benign inputs, but forces the model to produce an adversary-desired output if a specific trigger is present in the model input. Despite the initial success of current audio backdoor attacks, they suffer from the following limitations: (i) Most of them require sufficient knowledge, which limits their widespread adoption. (ii) They are not stealthy enough, thus easy to be detected by humans. (iii) Most of them cannot attack live speech, reducing their practicality. To address these problems, in this paper, we propose FlowMur, a stealthy and practical audio backdoor attack that can be launched with limited knowledge. FlowMur constructs an auxiliary dataset and a surrogate model to augment adversary knowledge. To achieve dynamicity, it formulates trigger generation as an optimization problem and optimizes the trigger over different attachment positions. To enhance stealthiness, we propose an adaptive data poisoning method according to Signal-to-Noise Ratio (SNR). Furthermore, ambient noise is incorporated into the process of trigger generation and data poisoning to make FlowMur robust to ambient noise and improve its practicality. Extensive experiments conducted on two datasets demonstrate that FlowMur achieves high attack performance in both digital and physical settings while remaining resilient to state-of-the-art defenses. In particular, a human study confirms that triggers generated by FlowMur are not easily detected by participants.



## **6. Unsupervised and Supervised learning by Dense Associative Memory under replica symmetry breaking**

cond-mat.dis-nn

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09638v1) [paper-pdf](http://arxiv.org/pdf/2312.09638v1)

**Authors**: Linda Albanese, Andrea Alessandrelli, Alessia Annibale, Adriano Barra

**Abstract**: Statistical mechanics of spin glasses is one of the main strands toward a comprehension of information processing by neural networks and learning machines. Tackling this approach, at the fairly standard replica symmetric level of description, recently Hebbian attractor networks with multi-node interactions (often called Dense Associative Memories) have been shown to outperform their classical pairwise counterparts in a number of tasks, from their robustness against adversarial attacks and their capability to work with prohibitively weak signals to their supra-linear storage capacities. Focusing on mathematical techniques more than computational aspects, in this paper we relax the replica symmetric assumption and we derive the one-step broken-replica-symmetry picture of supervised and unsupervised learning protocols for these Dense Associative Memories: a phase diagram in the space of the control parameters is achieved, independently, both via the Parisi's hierarchy within then replica trick as well as via the Guerra's telescope within the broken-replica interpolation. Further, an explicit analytical investigation is provided to deepen both the big-data and ground state limits of these networks as well as a proof that replica symmetry breaking does not alter the thresholds for learning and slightly increases the maximal storage capacity. Finally the De Almeida and Thouless line, depicting the onset of instability of a replica symmetric description, is also analytically derived highlighting how, crossed this boundary, the broken replica description should be preferred.



## **7. A Malware Classification Survey on Adversarial Attacks and Defences**

cs.CR

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09636v1) [paper-pdf](http://arxiv.org/pdf/2312.09636v1)

**Authors**: Mahesh Datta Sai Ponnuru, Likhitha Amasala, Tanu Sree Bhimavarapu, Guna Chaitanya Garikipati

**Abstract**: As the number and complexity of malware attacks continue to increase, there is an urgent need for effective malware detection systems. While deep learning models are effective at detecting malware, they are vulnerable to adversarial attacks. Attacks like this can create malicious files that are resistant to detection, creating a significant cybersecurity risk. Recent research has seen the development of several adversarial attack and response approaches aiming at strengthening deep learning models' resilience to such attacks. This survey study offers an in-depth look at current research in adversarial attack and defensive strategies for malware classification in cybersecurity. The methods are classified into four categories: generative models, feature-based approaches, ensemble methods, and hybrid tactics. The article outlines cutting-edge procedures within each area, assessing their benefits and drawbacks. Each topic presents cutting-edge approaches and explores their advantages and disadvantages. In addition, the study discusses the datasets and assessment criteria that are often utilized on this subject. Finally, it identifies open research difficulties and suggests future study options. This document is a significant resource for malware categorization and cyber security researchers and practitioners.



## **8. Understanding and Improving Adversarial Attacks on Latent Diffusion Model**

cs.CV

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2310.04687v2) [paper-pdf](http://arxiv.org/pdf/2310.04687v2)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Latent Diffusion Model (LDM) achieves state-of-the-art performances in image generation yet raising copyright and privacy concerns. Adversarial attacks on LDM are then born to protect unauthorized images from being used in LDM-driven few-shot generation. However, these attacks suffer from moderate performance and excessive computational cost, especially in GPU memory. In this paper, we propose an effective adversarial attack on LDM that shows superior performance against state-of-the-art few-shot generation pipeline of LDM, for example, LoRA. We implement the attack with memory efficiency by introducing several mechanisms and decrease the memory cost of the attack to less than 6GB, which allows individual users to run the attack on a majority of consumer GPUs. Our proposed attack can be a practical tool for people facing the copyright and privacy risk brought by LDM to protect themselves.



## **9. Towards Transferable Targeted 3D Adversarial Attack in the Physical World**

cs.CV

11 pages, 7 figures

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09558v1) [paper-pdf](http://arxiv.org/pdf/2312.09558v1)

**Authors**: Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, Xingxing Wei

**Abstract**: Compared with transferable untargeted attacks, transferable targeted adversarial attacks could specify the misclassification categories of adversarial samples, posing a greater threat to security-critical tasks. In the meanwhile, 3D adversarial samples, due to their potential of multi-view robustness, can more comprehensively identify weaknesses in existing deep learning systems, possessing great application value. However, the field of transferable targeted 3D adversarial attacks remains vacant. The goal of this work is to develop a more effective technique that could generate transferable targeted 3D adversarial examples, filling the gap in this field. To achieve this goal, we design a novel framework named TT3D that could rapidly reconstruct from few multi-view images into Transferable Targeted 3D textured meshes. While existing mesh-based texture optimization methods compute gradients in the high-dimensional mesh space and easily fall into local optima, leading to unsatisfactory transferability and distinct distortions, TT3D innovatively performs dual optimization towards both feature grid and Multi-layer Perceptron (MLP) parameters in the grid-based NeRF space, which significantly enhances black-box transferability while enjoying naturalness. Experimental results show that TT3D not only exhibits superior cross-model transferability but also maintains considerable adaptability across different renders and vision tasks. More importantly, we produce 3D adversarial examples with 3D printing techniques in the real world and verify their robust performance under various scenarios.



## **10. Embodied Adversarial Attack: A Dynamic Robust Physical Attack in Autonomous Driving**

cs.CV

10 pages, 7 figures

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09554v1) [paper-pdf](http://arxiv.org/pdf/2312.09554v1)

**Authors**: Yitong Sun, Yao Huang, Xingxing Wei

**Abstract**: As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in autonomous driving, their vulnerability to environmental changes has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. To enhance the robustness of physical adversarial attacks in the real world, instead of statically optimizing a robust adversarial example via an off-line training manner like the existing methods, this paper proposes a brand new robust adversarial attack framework: Embodied Adversarial Attack (EAA) from the perspective of dynamic adaptation, which aims to employ the paradigm of embodied intelligence: Perception-Decision-Control to dynamically adjust the optimal attack strategy according to the current situations in real time. For the perception module, given the challenge of needing simulation for the victim's viewpoint, EAA innovatively devises a Perspective Transformation Network to estimate the target's transformation from the attacker's perspective. For the decision and control module, EAA adopts the laser-a highly manipulable medium to implement physical attacks, and further trains an attack agent with reinforcement learning to make it capable of instantaneously determining the best attack strategy based on the perceived information. Finally, we apply our framework to the autonomous driving scenario. A variety of experiments verify the high effectiveness of our method under complex scenes.



## **11. Adversarial Robustness on Image Classification with $k$-means**

cs.LG

6 pages, 3 figures, 2 equations, 1 algorithm

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09533v1) [paper-pdf](http://arxiv.org/pdf/2312.09533v1)

**Authors**: Rollin Omari, Junae Kim, Paul Montague

**Abstract**: In this paper we explore the challenges and strategies for enhancing the robustness of $k$-means clustering algorithms against adversarial manipulations. We evaluate the vulnerability of clustering algorithms to adversarial attacks, emphasising the associated security risks. Our study investigates the impact of incremental attack strength on training, introduces the concept of transferability between supervised and unsupervised models, and highlights the sensitivity of unsupervised models to sample distributions. We additionally introduce and evaluate an adversarial training method that improves testing performance in adversarial scenarios, and we highlight the importance of various parameters in the proposed training method, such as continuous learning, centroid initialisation, and adversarial step-count.



## **12. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples**

cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09520v1) [paper-pdf](http://arxiv.org/pdf/2312.09520v1)

**Authors**: Chen Ma, Ningfei Wang, Qi Alfred Chen, Chao Shen

**Abstract**: In Autonomous Driving (AD), real-time perception is a critical component responsible for detecting surrounding objects to ensure safe driving. While researchers have extensively explored the integrity of AD perception due to its safety and security implications, the aspect of availability (real-time performance) or latency has received limited attention. Existing works on latency-based attack have focused mainly on object detection, i.e., a component in camera-based AD perception, overlooking the entire camera-based AD perception, which hinders them to achieve effective system-level effects, such as vehicle crashes. In this paper, we propose SlowTrack, a novel framework for generating adversarial attacks to increase the execution time of camera-based AD perception. We propose a novel two-stage attack strategy along with the three new loss function designs. Our evaluation is conducted on four popular camera-based AD perception pipelines, and the results demonstrate that SlowTrack significantly outperforms existing latency-based attacks while maintaining comparable imperceptibility levels. Furthermore, we perform the evaluation on Baidu Apollo, an industry-grade full-stack AD system, and LGSVL, a production-grade AD simulator, with two scenarios to compare the system-level effects of SlowTrack and existing attacks. Our evaluation results show that the system-level effects can be significantly improved, i.e., the vehicle crash rate of SlowTrack is around 95% on average while existing works only have around 30%.



## **13. Effective and Imperceptible Adversarial Textual Attack via Multi-objectivization**

cs.CL

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2111.01528v4) [paper-pdf](http://arxiv.org/pdf/2111.01528v4)

**Authors**: Shengcai Liu, Ning Lu, Wenjing Hong, Chao Qian, Ke Tang

**Abstract**: The field of adversarial textual attack has significantly grown over the last few years, where the commonly considered objective is to craft adversarial examples (AEs) that can successfully fool the target model. However, the imperceptibility of attacks, which is also essential for practical attackers, is often left out by previous studies. In consequence, the crafted AEs tend to have obvious structural and semantic differences from the original human-written text, making them easily perceptible. In this work, we advocate leveraging multi-objectivization to address such issue. Specifically, we reformulate the problem of crafting AEs as a multi-objective optimization problem, where the attack imperceptibility is considered as an auxiliary objective. Then, we propose a simple yet effective evolutionary algorithm, dubbed HydraText, to solve this problem. To the best of our knowledge, HydraText is currently the only approach that can be effectively applied to both score-based and decision-based attack settings. Exhaustive experiments involving 44237 instances demonstrate that HydraText consistently achieves competitive attack success rates and better attack imperceptibility than the recently proposed attack approaches. A human evaluation study also shows that the AEs crafted by HydraText are more indistinguishable from human-written text. Finally, these AEs exhibit good transferability and can bring notable robustness improvement to the target model by adversarial training.



## **14. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

cs.CR

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09494v1) [paper-pdf](http://arxiv.org/pdf/2312.09494v1)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.



## **15. Continual Adversarial Defense**

cs.CV

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09481v1) [paper-pdf](http://arxiv.org/pdf/2312.09481v1)

**Authors**: Qian Wang, Yaoyao Liu, Hefei Ling, Yingwei Li, Qihao Liu, Ping Li, Jiazhong Chen, Alan Yuille, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that can generalize to all types of attacks, including unseen ones, is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks used by many attackers. The defense system needs to upgrade itself by utilizing few-shot defense feedback and efficient memory. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accuracy on both clean and adversarial images. We leverage cutting-edge continual learning, few-shot learning, and ensemble learning techniques to qualify the principles. Experiments conducted on CIFAR-10 and ImageNet-100 validate the effectiveness of our approach against multiple stages of 10 modern adversarial attacks and significant improvements over 10 baseline methods. In particular, CAD is capable of quickly adapting with minimal feedback and a low cost of defense failure, while maintaining good performance against old attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.



## **16. Coevolutionary Algorithm for Building Robust Decision Trees under Minimax Regret**

cs.LG

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.09078v1) [paper-pdf](http://arxiv.org/pdf/2312.09078v1)

**Authors**: Adam Żychowski, Andrew Perrault, Jacek Mańdziuk

**Abstract**: In recent years, there has been growing interest in developing robust machine learning (ML) models that can withstand adversarial attacks, including one of the most widely adopted, efficient, and interpretable ML algorithms-decision trees (DTs). This paper proposes a novel coevolutionary algorithm (CoEvoRDT) designed to create robust DTs capable of handling noisy high-dimensional data in adversarial contexts. Motivated by the limitations of traditional DT algorithms, we leverage adaptive coevolution to allow DTs to evolve and learn from interactions with perturbed input data. CoEvoRDT alternately evolves competing populations of DTs and perturbed features, enabling construction of DTs with desired properties. CoEvoRDT is easily adaptable to various target metrics, allowing the use of tailored robustness criteria such as minimax regret. Furthermore, CoEvoRDT has potential to improve the results of other state-of-the-art methods by incorporating their outcomes (DTs they produce) into the initial population and optimize them in the process of coevolution. Inspired by the game theory, CoEvoRDT utilizes mixed Nash equilibrium to enhance convergence. The method is tested on 20 popular datasets and shows superior performance compared to 4 state-of-the-art algorithms. It outperformed all competing methods on 13 datasets with adversarial accuracy metrics, and on all 20 considered datasets with minimax regret. Strong experimental results and flexibility in choosing the error measure make CoEvoRDT a promising approach for constructing robust DTs in real-world applications.



## **17. Concealing Sensitive Samples against Gradient Leakage in Federated Learning**

cs.LG

Defence against model inversion attack in federated learning

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2209.05724v2) [paper-pdf](http://arxiv.org/pdf/2209.05724v2)

**Authors**: Jing Wu, Munawar Hayat, Mingyi Zhou, Mehrtash Harandi

**Abstract**: Federated Learning (FL) is a distributed learning paradigm that enhances users privacy by eliminating the need for clients to share raw, private data with the server. Despite the success, recent studies expose the vulnerability of FL to model inversion attacks, where adversaries reconstruct users private data via eavesdropping on the shared gradient information. We hypothesize that a key factor in the success of such attacks is the low entanglement among gradients per data within the batch during stochastic optimization. This creates a vulnerability that an adversary can exploit to reconstruct the sensitive data. Building upon this insight, we present a simple, yet effective defense strategy that obfuscates the gradients of the sensitive data with concealed samples. To achieve this, we propose synthesizing concealed samples to mimic the sensitive data at the gradient level while ensuring their visual dissimilarity from the actual sensitive data. Compared to the previous art, our empirical evaluations suggest that the proposed technique provides the strongest protection while simultaneously maintaining the FL performance.



## **18. DRAM-Locker: A General-Purpose DRAM Protection Mechanism against Adversarial DNN Weight Attacks**

cs.AR

7 pages. arXiv admin note: text overlap with arXiv:2305.08034

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.09027v1) [paper-pdf](http://arxiv.org/pdf/2312.09027v1)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Arman Roohi, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: In this work, we propose DRAM-Locker as a robust general-purpose defense mechanism that can protect DRAM against various adversarial Deep Neural Network (DNN) weight attacks affecting data or page tables. DRAM-Locker harnesses the capabilities of in-DRAM swapping combined with a lock-table to prevent attackers from singling out specific DRAM rows to safeguard DNN's weight parameters. Our results indicate that DRAM-Locker can deliver a high level of protection downgrading the performance of targeted weight attacks to a random attack level. Furthermore, the proposed defense mechanism demonstrates no reduction in accuracy when applied to CIFAR-10 and CIFAR-100. Importantly, DRAM-Locker does not necessitate any software retraining or result in extra hardware burden.



## **19. Amicable Aid: Perturbing Images to Improve Classification Performance**

cs.CV

ICASSP 2023

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2112.04720v4) [paper-pdf](http://arxiv.org/pdf/2112.04720v4)

**Authors**: Juyeop Kim, Jun-Ho Choi, Soobeom Jang, Jong-Seok Lee

**Abstract**: While adversarial perturbation of images to attack deep image classification models pose serious security concerns in practice, this paper suggests a novel paradigm where the concept of image perturbation can benefit classification performance, which we call amicable aid. We show that by taking the opposite search direction of perturbation, an image can be modified to yield higher classification confidence and even a misclassified image can be made correctly classified. This can be also achieved with a large amount of perturbation by which the image is made unrecognizable by human eyes. The mechanism of the amicable aid is explained in the viewpoint of the underlying natural image manifold. Furthermore, we investigate the universal amicable aid, i.e., a fixed perturbation can be applied to multiple images to improve their classification results. While it is challenging to find such perturbations, we show that making the decision boundary as perpendicular to the image manifold as possible via training with modified data is effective to obtain a model for which universal amicable perturbations are more easily found.



## **20. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08793v1) [paper-pdf](http://arxiv.org/pdf/2312.08793v1)

**Authors**: Tony T. Wang, Miles Wang, Kaivu Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .



## **21. AVA: Inconspicuous Attribute Variation-based Adversarial Attack bypassing DeepFake Detection**

cs.CV

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08675v1) [paper-pdf](http://arxiv.org/pdf/2312.08675v1)

**Authors**: Xiangtao Meng, Li Wang, Shanqing Guo, Lei Ju, Qingchuan Zhao

**Abstract**: While DeepFake applications are becoming popular in recent years, their abuses pose a serious privacy threat. Unfortunately, most related detection algorithms to mitigate the abuse issues are inherently vulnerable to adversarial attacks because they are built atop DNN-based classification models, and the literature has demonstrated that they could be bypassed by introducing pixel-level perturbations. Though corresponding mitigation has been proposed, we have identified a new attribute-variation-based adversarial attack (AVA) that perturbs the latent space via a combination of Gaussian prior and semantic discriminator to bypass such mitigation. It perturbs the semantics in the attribute space of DeepFake images, which are inconspicuous to human beings (e.g., mouth open) but can result in substantial differences in DeepFake detection. We evaluate our proposed AVA attack on nine state-of-the-art DeepFake detection algorithms and applications. The empirical results demonstrate that AVA attack defeats the state-of-the-art black box attacks against DeepFake detectors and achieves more than a 95% success rate on two commercial DeepFake detectors. Moreover, our human study indicates that AVA-generated DeepFake images are often imperceptible to humans, which presents huge security and privacy concerns.



## **22. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models**

cs.CR

Version 2 updates: Added comparison of three more evaluation methods  and their reliability check using human labeling. Added results for  jailbreaking Llama2 (individual behavior) and included complexity and  hyperparameter analysis. Revised objectives for prompt leaking. Other minor  changes made

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2310.15140v2) [paper-pdf](http://arxiv.org/pdf/2310.15140v2)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent studies suggest that defending against these attacks is possible: adversarial attacks generate unlimited but unreadable gibberish prompts, detectable by perplexity-based filters; manual jailbreak attacks craft readable prompts, but their limited number due to the necessity of human creativity allows for easy blocking. In this paper, we show that these solutions may be too optimistic. We introduce AutoDAN, an interpretable, gradient-based adversarial attack that merges the strengths of both attack types. Guided by the dual goals of jailbreak and readability, AutoDAN optimizes and generates tokens one by one from left to right, resulting in readable prompts that bypass perplexity filters while maintaining high attack success rates. Notably, these prompts, generated from scratch using gradients, are interpretable and diverse, with emerging strategies commonly seen in manual jailbreak attacks. They also generalize to unforeseen harmful behaviors and transfer to black-box LLMs better than their unreadable counterparts when using limited training data or a single proxy model. Furthermore, we show the versatility of AutoDAN by automatically leaking system prompts using a customized objective. Our work offers a new way to red-team LLMs and understand jailbreak mechanisms via interpretability.



## **23. Towards Inductive Robustness: Distilling and Fostering Wave-induced Resonance in Transductive GCNs Against Graph Adversarial Attacks**

cs.LG

AAAI 2024

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08651v1) [paper-pdf](http://arxiv.org/pdf/2312.08651v1)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Pan Zhou

**Abstract**: Graph neural networks (GNNs) have recently been shown to be vulnerable to adversarial attacks, where slight perturbations in the graph structure can lead to erroneous predictions. However, current robust models for defending against such attacks inherit the transductive limitations of graph convolutional networks (GCNs). As a result, they are constrained by fixed structures and do not naturally generalize to unseen nodes. Here, we discover that transductive GCNs inherently possess a distillable robustness, achieved through a wave-induced resonance process. Based on this, we foster this resonance to facilitate inductive and robust learning. Specifically, we first prove that the signal formed by GCN-driven message passing (MP) is equivalent to the edge-based Laplacian wave, where, within a wave system, resonance can naturally emerge between the signal and its transmitting medium. This resonance provides inherent resistance to malicious perturbations inflicted on the signal system. We then prove that merely three MP iterations within GCNs can induce signal resonance between nodes and edges, manifesting as a coupling between nodes and their distillable surrounding local subgraph. Consequently, we present Graph Resonance-fostering Network (GRN) to foster this resonance via learning node representations from their distilled resonating subgraphs. By capturing the edge-transmitted signals within this subgraph and integrating them with the node signal, GRN embeds these combined signals into the central node's representation. This node-wise embedding approach allows for generalization to unseen nodes. We validate our theoretical findings with experiments, and demonstrate that GRN generalizes robustness to unseen nodes, whilst maintaining state-of-the-art classification accuracy on perturbed graphs.



## **24. Guarding the Grid: Enhancing Resilience in Automated Residential Demand Response Against False Data Injection Attacks**

eess.SY

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08646v1) [paper-pdf](http://arxiv.org/pdf/2312.08646v1)

**Authors**: Thusitha Dayaratne, Carsten Rudolph, Ariel Liebman, Mahsa Salehi

**Abstract**: Utility companies are increasingly leveraging residential demand flexibility and the proliferation of smart/IoT devices to enhance the effectiveness of residential demand response (DR) programs through automated device scheduling. However, the adoption of distributed architectures in these systems exposes them to the risk of false data injection attacks (FDIAs), where adversaries can manipulate decision-making processes by injecting false data. Given the limited control utility companies have over these distributed systems and data, the need for reliable implementations to enhance the resilience of residential DR schemes against FDIAs is paramount. In this work, we present a comprehensive framework that combines DR optimisation, anomaly detection, and strategies for mitigating the impacts of attacks to create a resilient and automated device scheduling system. To validate the robustness of our framework against FDIAs, we performed an evaluation using real-world data sets, highlighting its effectiveness in securing residential DR systems.



## **25. Scalable Ensemble-based Detection Method against Adversarial Attacks for speaker verification**

eess.AS

Submitted to 2024 ICASSP

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08622v1) [paper-pdf](http://arxiv.org/pdf/2312.08622v1)

**Authors**: Haibin Wu, Heng-Cheng Kuo, Yu Tsao, Hung-yi Lee

**Abstract**: Automatic speaker verification (ASV) is highly susceptible to adversarial attacks. Purification modules are usually adopted as a pre-processing to mitigate adversarial noise. However, they are commonly implemented across diverse experimental settings, rendering direct comparisons challenging. This paper comprehensively compares mainstream purification techniques in a unified framework. We find these methods often face a trade-off between user experience and security, as they struggle to simultaneously maintain genuine sample performance and reduce adversarial perturbations. To address this challenge, some efforts have extended purification modules to encompass detection capabilities, aiming to alleviate the trade-off. However, advanced purification modules will always come into the stage to surpass previous detection method. As a result, we further propose an easy-to-follow ensemble approach that integrates advanced purification modules for detection, achieving state-of-the-art (SOTA) performance in countering adversarial noise. Our ensemble method has great potential due to its compatibility with future advanced purification techniques.



## **26. Exploring the Privacy Risks of Adversarial VR Game Design**

cs.CR

Learn more at https://rdi.berkeley.edu/metaverse/metadata

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2207.13176v4) [paper-pdf](http://arxiv.org/pdf/2207.13176v4)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song, James F. O'Brien

**Abstract**: Fifty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Within just a few minutes, an adversarial program had accurately inferred over 25 of their personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. Since the Cambridge Analytica scandal of 2018, adversarially designed gamified elements have been known to constitute a significant privacy threat in conventional social platforms. In this work, we present a case study of how metaverse environments can similarly be adversarially constructed to covertly infer dozens of personal data attributes from seemingly anonymous users. While existing VR privacy research largely focuses on passive observation, we argue that because individuals subconsciously reveal personal information via their motion in response to specific stimuli, active attacks pose an outsized risk in VR environments.



## **27. Defenses in Adversarial Machine Learning: A Survey**

cs.CV

21 pages, 5 figures, 2 tables, 237 reference papers

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08890v1) [paper-pdf](http://arxiv.org/pdf/2312.08890v1)

**Authors**: Baoyuan Wu, Shaokui Wei, Mingli Zhu, Meixi Zheng, Zihao Zhu, Mingda Zhang, Hongrui Chen, Danni Yuan, Li Liu, Qingshan Liu

**Abstract**: Adversarial phenomenon has been widely observed in machine learning (ML) systems, especially in those using deep neural networks, describing that ML systems may produce inconsistent and incomprehensible predictions with humans at some particular cases. This phenomenon poses a serious security threat to the practical application of ML systems, and several advanced attack paradigms have been developed to explore it, mainly including backdoor attacks, weight attacks, and adversarial examples. For each individual attack paradigm, various defense paradigms have been developed to improve the model robustness against the corresponding attack paradigm. However, due to the independence and diversity of these defense paradigms, it is difficult to examine the overall robustness of an ML system against different kinds of attacks.This survey aims to build a systematic review of all existing defense paradigms from a unified perspective. Specifically, from the life-cycle perspective, we factorize a complete machine learning system into five stages, including pre-training, training, post-training, deployment, and inference stages, respectively. Then, we present a clear taxonomy to categorize and review representative defense methods at each individual stage. The unified perspective and presented taxonomies not only facilitate the analysis of the mechanism of each defense paradigm but also help us to understand connections and differences among different defense paradigms, which may inspire future research to develop more advanced, comprehensive defenses.



## **28. Universal Adversarial Framework to Improve Adversarial Robustness for Diabetic Retinopathy Detection**

eess.IV

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08193v1) [paper-pdf](http://arxiv.org/pdf/2312.08193v1)

**Authors**: Samrat Mukherjee, Dibyanayan Bandyopadhyay, Baban Gain, Asif Ekbal

**Abstract**: Diabetic Retinopathy (DR) is a prevalent illness associated with Diabetes which, if left untreated, can result in irreversible blindness. Deep Learning based systems are gradually being introduced as automated support for clinical diagnosis. Since healthcare has always been an extremely important domain demanding error-free performance, any adversaries could pose a big threat to the applicability of such systems. In this work, we use Universal Adversarial Perturbations (UAPs) to quantify the vulnerability of Medical Deep Neural Networks (DNNs) for detecting DR. To the best of our knowledge, this is the very first attempt that works on attacking complete fine-grained classification of DR images using various UAPs. Also, as a part of this work, we use UAPs to fine-tune the trained models to defend against adversarial samples. We experiment on several models and observe that the performance of such models towards unseen adversarial attacks gets boosted on average by $3.41$ Cohen-kappa value and maximum by $31.92$ Cohen-kappa value. The performance degradation on normal data upon ensembling the fine-tuned models was found to be statistically insignificant using t-test, highlighting the benefits of UAP-based adversarial fine-tuning.



## **29. Adversarial Attacks on Graph Neural Networks based Spatial Resource Management in P2P Wireless Communications**

eess.SP

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08181v1) [paper-pdf](http://arxiv.org/pdf/2312.08181v1)

**Authors**: Ahmad Ghasemi, Ehsan Zeraatkar, Majid Moradikia, Seyed, Zekavat

**Abstract**: This paper introduces adversarial attacks targeting a Graph Neural Network (GNN) based radio resource management system in point to point (P2P) communications. Our focus lies on perturbing the trained GNN model during the test phase, specifically targeting its vertices and edges. To achieve this, four distinct adversarial attacks are proposed, each accounting for different constraints, and aiming to manipulate the behavior of the system. The proposed adversarial attacks are formulated as optimization problems, aiming to minimize the system's communication quality. The efficacy of these attacks is investigated against the number of users, signal-to-noise ratio (SNR), and adversary power budget. Furthermore, we address the detection of such attacks from the perspective of the Central Processing Unit (CPU) of the system. To this end, we formulate an optimization problem that involves analyzing the distribution of channel eigenvalues before and after the attacks are applied. This formulation results in a Min-Max optimization problem, allowing us to detect the presence of attacks. Through extensive simulations, we observe that in the absence of adversarial attacks, the eigenvalues conform to Johnson's SU distribution. However, the attacks significantly alter the characteristics of the eigenvalue distribution, and in the most effective attack, they even change the type of the eigenvalue distribution.



## **30. Efficient Representation of the Activation Space in Deep Neural Networks**

cs.LG

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08143v1) [paper-pdf](http://arxiv.org/pdf/2312.08143v1)

**Authors**: Tanya Akumu, Celia Cintas, Girmaw Abebe Tadesse, Adebayo Oshingbesan, Skyler Speakman, Edward McFowland III

**Abstract**: The representations of the activation space of deep neural networks (DNNs) are widely utilized for tasks like natural language processing, anomaly detection and speech recognition. Due to the diverse nature of these tasks and the large size of DNNs, an efficient and task-independent representation of activations becomes crucial. Empirical p-values have been used to quantify the relative strength of an observed node activation compared to activations created by already-known inputs. Nonetheless, keeping raw data for these calculations increases memory resource consumption and raises privacy concerns. To this end, we propose a model-agnostic framework for creating representations of activations in DNNs using node-specific histograms to compute p-values of observed activations without retaining already-known inputs. Our proposed approach demonstrates promising potential when validated with multiple network architectures across various downstream tasks and compared with the kernel density estimates and brute-force empirical baselines. In addition, the framework reduces memory usage by 30% with up to 4 times faster p-value computing time while maintaining state of-the-art detection power in downstream tasks such as the detection of adversarial attacks and synthesized content. Moreover, as we do not persist raw data at inference time, we could potentially reduce susceptibility to attacks and privacy issues.



## **31. Robust Few-Shot Named Entity Recognition with Boundary Discrimination and Correlation Purification**

cs.CL

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07961v1) [paper-pdf](http://arxiv.org/pdf/2312.07961v1)

**Authors**: Xiaojun Xue, Chunxia Zhang, Tianxiang Xu, Zhendong Niu

**Abstract**: Few-shot named entity recognition (NER) aims to recognize novel named entities in low-resource domains utilizing existing knowledge. However, the present few-shot NER models assume that the labeled data are all clean without noise or outliers, and there are few works focusing on the robustness of the cross-domain transfer learning ability to textual adversarial attacks in Few-shot NER. In this work, we comprehensively explore and assess the robustness of few-shot NER models under textual adversarial attack scenario, and found the vulnerability of existing few-shot NER models. Furthermore, we propose a robust two-stage few-shot NER method with Boundary Discrimination and Correlation Purification (BDCP). Specifically, in the span detection stage, the entity boundary discriminative module is introduced to provide a highly distinguishing boundary representation space to detect entity spans. In the entity typing stage, the correlations between entities and contexts are purified by minimizing the interference information and facilitating correlation generalization to alleviate the perturbations caused by textual adversarial attacks. In addition, we construct adversarial examples for few-shot NER based on public datasets Few-NERD and Cross-Dataset. Comprehensive evaluations on those two groups of few-shot NER datasets containing adversarial examples demonstrate the robustness and superiority of the proposed method.



## **32. DifAttack: Query-Efficient Black-Box Attack via Disentangled Feature Space**

cs.CV

Accepted in AAAI'24

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2309.14585v3) [paper-pdf](http://arxiv.org/pdf/2309.14585v3)

**Authors**: Liu Jun, Zhou Jiantao, Zeng Jiandian, Jinyu Tian

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (ASR) and good generalizability. We design a novel attack method based on a Disentangled Feature space, called DifAttack, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack firstly disentangles an image's latent feature into an adversarial feature and a visual feature, where the former dominates the adversarial capability of an image, while the latter largely determines its visual appearance. We train an autoencoder for the disentanglement by using pairs of clean images and their Adversarial Examples (AEs) generated from available surrogate models via white-box attack methods. Eventually, DifAttack iteratively optimizes the adversarial feature according to the query feedback from the victim model until a successful AE is generated, while keeping the visual feature unaltered. In addition, due to the avoidance of using surrogate models' gradient information when optimizing AEs for black-box models, our proposed DifAttack inherently possesses better attack capability in the open-set scenario, where the training dataset of the victim model is unknown. Extensive experimental results demonstrate that our method achieves significant improvements in ASR and query efficiency simultaneously, especially in the targeted attack and open-set scenarios. The code is available at https://github.com/csjunjun/DifAttack.git.



## **33. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07910v1) [paper-pdf](http://arxiv.org/pdf/2312.07910v1)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **34. Causality Analysis for Evaluating the Security of Large Language Models**

cs.AI

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07876v1) [paper-pdf](http://arxiv.org/pdf/2312.07876v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.



## **35. Securing Graph Neural Networks in MLaaS: A Comprehensive Realization of Query-based Integrity Verification**

cs.CR

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07870v1) [paper-pdf](http://arxiv.org/pdf/2312.07870v1)

**Authors**: Bang Wu, Xingliang Yuan, Shuo Wang, Qi Li, Minhui Xue, Shirui Pan

**Abstract**: The deployment of Graph Neural Networks (GNNs) within Machine Learning as a Service (MLaaS) has opened up new attack surfaces and an escalation in security concerns regarding model-centric attacks. These attacks can directly manipulate the GNN model parameters during serving, causing incorrect predictions and posing substantial threats to essential GNN applications. Traditional integrity verification methods falter in this context due to the limitations imposed by MLaaS and the distinct characteristics of GNN models.   In this research, we introduce a groundbreaking approach to protect GNN models in MLaaS from model-centric attacks. Our approach includes a comprehensive verification schema for GNN's integrity, taking into account both transductive and inductive GNNs, and accommodating varying pre-deployment knowledge of the models. We propose a query-based verification technique, fortified with innovative node fingerprint generation algorithms. To deal with advanced attackers who know our mechanisms in advance, we introduce randomized fingerprint nodes within our design. The experimental evaluation demonstrates that our method can detect five representative adversarial model-centric attacks, displaying 2 to 4 times greater efficiency compared to baselines.



## **36. SimAC: A Simple Anti-Customization Method against Text-to-Image Synthesis of Diffusion Models**

cs.CV

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07865v1) [paper-pdf](http://arxiv.org/pdf/2312.07865v1)

**Authors**: Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, Qidong Huang

**Abstract**: Despite the success of diffusion-based customization methods on visual content creation, increasing concerns have been raised about such techniques from both privacy and political perspectives. To tackle this issue, several anti-customization methods have been proposed in very recent months, predominantly grounded in adversarial attacks. Unfortunately, most of these methods adopt straightforward designs, such as end-to-end optimization with a focus on adversarially maximizing the original training loss, thereby neglecting nuanced internal properties intrinsic to the diffusion model, and even leading to ineffective optimization in some diffusion time steps. In this paper, we strive to bridge this gap by undertaking a comprehensive exploration of these inherent properties, to boost the performance of current anti-customization approaches. Two aspects of properties are investigated: 1) We examine the relationship between time step selection and the model's perception in the frequency domain of images and find that lower time steps can give much more contributions to adversarial noises. This inspires us to propose an adaptive greedy search for optimal time steps that seamlessly integrates with existing anti-customization methods. 2) We scrutinize the roles of features at different layers during denoising and devise a sophisticated feature-based optimization framework for anti-customization. Experiments on facial benchmarks demonstrate that our approach significantly increases identity disruption, thereby enhancing user privacy and security.



## **37. Radio Signal Classification by Adversarially Robust Quantum Machine Learning**

quant-ph

12 pages, 6 figures

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07821v1) [paper-pdf](http://arxiv.org/pdf/2312.07821v1)

**Authors**: Yanqiu Wu, Eromanga Adermann, Chandra Thapa, Seyit Camtepe, Hajime Suzuki, Muhammad Usman

**Abstract**: Radio signal classification plays a pivotal role in identifying the modulation scheme used in received radio signals, which is essential for demodulation and proper interpretation of the transmitted information. Researchers have underscored the high susceptibility of ML algorithms for radio signal classification to adversarial attacks. Such vulnerability could result in severe consequences, including misinterpretation of critical messages, interception of classified information, or disruption of communication channels. Recent advancements in quantum computing have revolutionized theories and implementations of computation, bringing the unprecedented development of Quantum Machine Learning (QML). It is shown that quantum variational classifiers (QVCs) provide notably enhanced robustness against classical adversarial attacks in image classification. However, no research has yet explored whether QML can similarly mitigate adversarial threats in the context of radio signal classification. This work applies QVCs to radio signal classification and studies their robustness to various adversarial attacks. We also propose the novel application of the approximate amplitude encoding (AAE) technique to encode radio signal data efficiently. Our extensive simulation results present that attacks generated on QVCs transfer well to CNN models, indicating that these adversarial examples can fool neural networks that they are not explicitly designed to attack. However, the converse is not true. QVCs primarily resist the attacks generated on CNNs. Overall, with comprehensive simulations, our results shed new light on the growing field of QML by bridging knowledge gaps in QAML in radio signal classification and uncovering the advantages of applying QML methods in practical applications.



## **38. BarraCUDA: Bringing Electromagnetic Side Channel Into Play to Steal the Weights of Neural Networks from NVIDIA GPUs**

cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07783v1) [paper-pdf](http://arxiv.org/pdf/2312.07783v1)

**Authors**: Peter Horvath, Lukasz Chmielewski, Leo Weissbart, Lejla Batina, Yuval Yarom

**Abstract**: Over the last decade, applications of neural networks have spread to cover all aspects of life. A large number of companies base their businesses on building products that use neural networks for tasks such as face recognition, machine translation, and autonomous cars. They are being used in safety and security-critical applications like high definition maps and medical wristbands, or in globally used products like Google Translate and ChatGPT. Much of the intellectual property underpinning these products is encoded in the exact configuration of the neural networks. Consequently, protecting these is of utmost priority to businesses. At the same time, many of these products need to operate under a strong threat model, in which the adversary has unfettered physical control of the product.   Past work has demonstrated that with physical access, attackers can reverse engineer neural networks that run on scalar microcontrollers, like ARM Cortex M3. However, for performance reasons, neural networks are often implemented on highly-parallel general purpose graphics processing units (GPGPUs), and so far, attacks on these have only recovered course-grained information on the structure of the neural network, but failed to retrieve the weights and biases.   In this work, we present BarraCUDA, a novel attack on GPGPUs that can completely extract the parameters of neural networks. BarraCUDA uses correlation electromagnetic analysis to recover the weights and biases in the convolutional layers of neural networks. We use BarraCUDA to attack the popular NVIDIA Jetson Nano device, demonstrating successful parameter extraction of neural networks in a highly parallel and noisy environment.



## **39. Majority is Not Required: A Rational Analysis of the Private Double-Spend Attack from a Sub-Majority Adversary**

cs.GT

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07709v1) [paper-pdf](http://arxiv.org/pdf/2312.07709v1)

**Authors**: Yanni Georghiades, Rajesh Mishra, Karl Kreder, Sriram Vishwanath

**Abstract**: We study the incentives behind double-spend attacks on Nakamoto-style Proof-of-Work cryptocurrencies. In these systems, miners are allowed to choose which transactions to reference with their block, and a common strategy for selecting transactions is to simply choose those with the highest fees. This can be problematic if these transactions originate from an adversary with substantial (but less than 50\%) computational power, as high-value transactions can present an incentive for a rational adversary to attempt a double-spend attack if they expect to profit. The most common mechanism for deterring double-spend attacks is for the recipients of large transactions to wait for additional block confirmations (i.e., to increase the attack cost). We argue that this defense mechanism is not satisfactory, as the security of the system is contingent on the actions of its users. Instead, we propose that defending against double-spend attacks should be the responsibility of the miners; specifically, miners should limit the amount of transaction value they include in a block (i.e., reduce the attack reward). To this end, we model cryptocurrency mining as a mean-field game in which we augment the standard mining reward function to simulate the presence of a rational, double-spending adversary. We design and implement an algorithm which characterizes the behavior of miners at equilibrium, and we show that miners who use the adversary-aware reward function accumulate more wealth than those who do not. We show that the optimal strategy for honest miners is to limit the amount of value transferred by each block such that the adversary's expected profit is 0. Additionally, we examine Bitcoin's resilience to double-spend attacks. Assuming a 6 block confirmation time, we find that an attacker with at least 25% of the network mining power can expect to profit from a double-spend attack.



## **40. Defending Our Privacy With Backdoors**

cs.LG

14 pages, 10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2310.08320v2) [paper-pdf](http://arxiv.org/pdf/2310.08320v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspective on backdoor attacks, but also presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.



## **41. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.04730v2) [paper-pdf](http://arxiv.org/pdf/2312.04730v2)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.



## **42. ReRoGCRL: Representation-based Robustness in Goal-Conditioned Reinforcement Learning**

cs.LG

This paper has been accepted in AAAI24  (https://aaai.org/aaai-conference/)

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07392v1) [paper-pdf](http://arxiv.org/pdf/2312.07392v1)

**Authors**: Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan

**Abstract**: While Goal-Conditioned Reinforcement Learning (GCRL) has gained attention, its algorithmic robustness, particularly against adversarial perturbations, remains unexplored. Unfortunately, the attacks and robust representation training methods specifically designed for traditional RL are not so effective when applied to GCRL. To address this challenge, we propose the \textit{Semi-Contrastive Representation} attack, a novel approach inspired by the adversarial contrastive attack. Unlike existing attacks in RL, it only necessitates information from the policy function and can be seamlessly implemented during deployment. Furthermore, to mitigate the vulnerability of existing GCRL algorithms, we introduce \textit{Adversarial Representation Tactics}. This strategy combines \textit{Semi-Contrastive Adversarial Augmentation} with \textit{Sensitivity-Aware Regularizer}. It improves the adversarial robustness of the underlying agent against various types of perturbations. Extensive experiments validate the superior performance of our attack and defence mechanism across multiple state-of-the-art GCRL algorithms. Our tool {\bf ReRoGCRL} is available at \url{https://github.com/TrustAI/ReRoGCRL}.



## **43. Eroding Trust In Aerial Imagery: Comprehensive Analysis and Evaluation Of Adversarial Attacks In Geospatial Systems**

cs.CV

Accepted at IEEE AIRP 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07389v1) [paper-pdf](http://arxiv.org/pdf/2312.07389v1)

**Authors**: Michael Lanier, Aayush Dhakal, Zhexiao Xiong, Arthur Li, Nathan Jacobs, Yevgeniy Vorobeychik

**Abstract**: In critical operations where aerial imagery plays an essential role, the integrity and trustworthiness of data are paramount. The emergence of adversarial attacks, particularly those that exploit control over labels or employ physically feasible trojans, threatens to erode that trust, making the analysis and mitigation of these attacks a matter of urgency. We demonstrate how adversarial attacks can degrade confidence in geospatial systems, specifically focusing on scenarios where the attacker's control over labels is restricted and the use of realistic threat vectors. Proposing and evaluating several innovative attack methodologies, including those tailored to overhead images, we empirically show their threat to remote sensing systems using high-quality SpaceNet datasets. Our experimentation reflects the unique challenges posed by aerial imagery, and these preliminary results not only reveal the potential risks but also highlight the non-trivial nature of the problem compared to recent works.



## **44. SSTA: Salient Spatially Transformed Attack**

cs.CV

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07258v1) [paper-pdf](http://arxiv.org/pdf/2312.07258v1)

**Authors**: Renyang Liu, Wei Zhou, Sixin Wu, Jun Zhao, Kwok-Yan Lam

**Abstract**: Extensive studies have demonstrated that deep neural networks (DNNs) are vulnerable to adversarial attacks, which brings a huge security risk to the further application of DNNs, especially for the AI models developed in the real world. Despite the significant progress that has been made recently, existing attack methods still suffer from the unsatisfactory performance of escaping from being detected by naked human eyes due to the formulation of adversarial example (AE) heavily relying on a noise-adding manner. Such mentioned challenges will significantly increase the risk of exposure and result in an attack to be failed. Therefore, in this paper, we propose the Salient Spatially Transformed Attack (SSTA), a novel framework to craft imperceptible AEs, which enhance the stealthiness of AEs by estimating a smooth spatial transform metric on a most critical area to generate AEs instead of adding external noise to the whole image. Compared to state-of-the-art baselines, extensive experiments indicated that SSTA could effectively improve the imperceptibility of the AEs while maintaining a 100\% attack success rate.



## **45. DTA: Distribution Transform-based Attack for Query-Limited Scenario**

cs.CV

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07245v1) [paper-pdf](http://arxiv.org/pdf/2312.07245v1)

**Authors**: Renyang Liu, Wei Zhou, Xin Jin, Song Gao, Yuanyu Wang, Ruxin Wang

**Abstract**: In generating adversarial examples, the conventional black-box attack methods rely on sufficient feedback from the to-be-attacked models by repeatedly querying until the attack is successful, which usually results in thousands of trials during an attack. This may be unacceptable in real applications since Machine Learning as a Service Platform (MLaaS) usually only returns the final result (i.e., hard-label) to the client and a system equipped with certain defense mechanisms could easily detect malicious queries. By contrast, a feasible way is a hard-label attack that simulates an attacked action being permitted to conduct a limited number of queries. To implement this idea, in this paper, we bypass the dependency on the to-be-attacked model and benefit from the characteristics of the distributions of adversarial examples to reformulate the attack problem in a distribution transform manner and propose a distribution transform-based attack (DTA). DTA builds a statistical mapping from the benign example to its adversarial counterparts by tackling the conditional likelihood under the hard-label black-box settings. In this way, it is no longer necessary to query the target model frequently. A well-trained DTA model can directly and efficiently generate a batch of adversarial examples for a certain input, which can be used to attack un-seen models based on the assumed transferability. Furthermore, we surprisingly find that the well-trained DTA model is not sensitive to the semantic spaces of the training dataset, meaning that the model yields acceptable attack performance on other datasets. Extensive experiments validate the effectiveness of the proposed idea and the superiority of DTA over the state-of-the-art.



## **46. Reward Certification for Policy Smoothed Reinforcement Learning**

cs.LG

This paper will be presented in AAAI2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06436v2) [paper-pdf](http://arxiv.org/pdf/2312.06436v2)

**Authors**: Ronghui Mu, Leandro Soriano Marcolino, Tianle Zhang, Yanghao Zhang, Xiaowei Huang, Wenjie Ruan

**Abstract**: Reinforcement Learning (RL) has achieved remarkable success in safety-critical areas, but it can be weakened by adversarial attacks. Recent studies have introduced "smoothed policies" in order to enhance its robustness. Yet, it is still challenging to establish a provable guarantee to certify the bound of its total reward. Prior methods relied primarily on computing bounds using Lipschitz continuity or calculating the probability of cumulative reward above specific thresholds. However, these techniques are only suited for continuous perturbations on the RL agent's observations and are restricted to perturbations bounded by the $l_2$-norm. To address these limitations, this paper proposes a general black-box certification method capable of directly certifying the cumulative reward of the smoothed policy under various $l_p$-norm bounded perturbations. Furthermore, we extend our methodology to certify perturbations on action spaces. Our approach leverages f-divergence to measure the distinction between the original distribution and the perturbed distribution, subsequently determining the certification bound by solving a convex optimisation problem. We provide a comprehensive theoretical analysis and run sufficient experiments in multiple environments. Our results show that our method not only improves the certified lower bound of mean cumulative reward but also demonstrates better efficiency than state-of-the-art techniques.



## **47. Adversarial Driving: Attacking End-to-End Autonomous Driving**

cs.CV

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2103.09151v8) [paper-pdf](http://arxiv.org/pdf/2103.09151v8)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: As research in deep neural networks advances, deep convolutional networks become promising for autonomous driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for autonomous driving. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. While for regression tasks, the effect of adversarial attacks is not as well understood. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving models. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. In an average of 800 attacks with the same attack strength (epsilon=1), the image-specific and image-agnostic attack deviates the steering angle from the original output by 0.478 and 0.111, respectively, which is much stronger than random noises that only perturbs the steering angle by 0.002 (The steering angle ranges from [-1, 1]). Both attacks can be initiated in real-time on CPUs without employing GPUs. Demo video: https://youtu.be/I0i8uN2oOP0.



## **48. Adversarial Detection: Attacking Object Detection in Real Time**

cs.AI

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2209.01962v6) [paper-pdf](http://arxiv.org/pdf/2209.01962v6)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90% within about 20 iterations. The demo video is available at https://youtu.be/zJZ1aNlXsMU.



## **49. Cost Aware Untargeted Poisoning Attack against Graph Neural Networks,**

cs.AI

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07158v1) [paper-pdf](http://arxiv.org/pdf/2312.07158v1)

**Authors**: Yuwei Han, Yuni Lai, Yulin Zhu, Kai Zhou

**Abstract**: Graph Neural Networks (GNNs) have become widely used in the field of graph mining. However, these networks are vulnerable to structural perturbations. While many research efforts have focused on analyzing vulnerability through poisoning attacks, we have identified an inefficiency in current attack losses. These losses steer the attack strategy towards modifying edges targeting misclassified nodes or resilient nodes, resulting in a waste of structural adversarial perturbation. To address this issue, we propose a novel attack loss framework called the Cost Aware Poisoning Attack (CA-attack) to improve the allocation of the attack budget by dynamically considering the classification margins of nodes. Specifically, it prioritizes nodes with smaller positive margins while postponing nodes with negative margins. Our experiments demonstrate that the proposed CA-attack significantly enhances existing attack strategies



## **50. Data-Free Hard-Label Robustness Stealing Attack**

cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.05924v2) [paper-pdf](http://arxiv.org/pdf/2312.05924v2)

**Authors**: Xiaojian Yuan, Kejiang Chen, Wen Huang, Jie Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: The popularity of Machine Learning as a Service (MLaaS) has led to increased concerns about Model Stealing Attacks (MSA), which aim to craft a clone model by querying MLaaS. Currently, most research on MSA assumes that MLaaS can provide soft labels and that the attacker has a proxy dataset with a similar distribution. However, this fails to encapsulate the more practical scenario where only hard labels are returned by MLaaS and the data distribution remains elusive. Furthermore, most existing work focuses solely on stealing the model accuracy, neglecting the model robustness, while robustness is essential in security-sensitive scenarios, e.g., face-scan payment. Notably, improving model robustness often necessitates the use of expensive techniques such as adversarial training, thereby further making stealing robustness a more lucrative prospect. In response to these identified gaps, we introduce a novel Data-Free Hard-Label Robustness Stealing (DFHL-RS) attack in this paper, which enables the stealing of both model accuracy and robustness by simply querying hard labels of the target model without the help of any natural data. Comprehensive experiments demonstrate the effectiveness of our method. The clone model achieves a clean accuracy of 77.86% and a robust accuracy of 39.51% against AutoAttack, which are only 4.71% and 8.40% lower than the target model on the CIFAR-10 dataset, significantly exceeding the baselines. Our code is available at: https://github.com/LetheSec/DFHL-RS-Attack.



