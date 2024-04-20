# Latest Adversarial Attack Papers
**update at 2024-04-20 09:32:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. KDk: A Defense Mechanism Against Label Inference Attacks in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12369v1) [paper-pdf](http://arxiv.org/pdf/2404.12369v1)

**Authors**: Marco Arazzi, Serena Nicolazzo, Antonino Nocera

**Abstract**: Vertical Federated Learning (VFL) is a category of Federated Learning in which models are trained collaboratively among parties with vertically partitioned data. Typically, in a VFL scenario, the labels of the samples are kept private from all the parties except for the aggregating server, that is the label owner. Nevertheless, recent works discovered that by exploiting gradient information returned by the server to bottom models, with the knowledge of only a small set of auxiliary labels on a very limited subset of training data points, an adversary can infer the private labels. These attacks are known as label inference attacks in VFL. In our work, we propose a novel framework called KDk, that combines Knowledge Distillation and k-anonymity to provide a defense mechanism against potential label inference attacks in a VFL scenario. Through an exhaustive experimental campaign we demonstrate that by applying our approach, the performance of the analyzed label inference attacks decreases consistently, even by more than 60%, maintaining the accuracy of the whole VFL almost unaltered.



## **2. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.03027v2) [paper-pdf](http://arxiv.org/pdf/2404.03027v2)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.



## **3. Transferability Ranking of Adversarial Examples**

cs.LG

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2208.10878v2) [paper-pdf](http://arxiv.org/pdf/2208.10878v2)

**Authors**: Mosh Levy, Guy Amit, Yuval Elovici, Yisroel Mirsky

**Abstract**: Adversarial transferability in black-box scenarios presents a unique challenge: while attackers can employ surrogate models to craft adversarial examples, they lack assurance on whether these examples will successfully compromise the target model. Until now, the prevalent method to ascertain success has been trial and error-testing crafted samples directly on the victim model. This approach, however, risks detection with every attempt, forcing attackers to either perfect their first try or face exposure. Our paper introduces a ranking strategy that refines the transfer attack process, enabling the attacker to estimate the likelihood of success without repeated trials on the victim's system. By leveraging a set of diverse surrogate models, our method can predict transferability of adversarial examples. This strategy can be used to either select the best sample to use in an attack or the best perturbation to apply to a specific sample. Using our strategy, we were able to raise the transferability of adversarial examples from a mere 20% - akin to random selection-up to near upper-bound levels, with some scenarios even witnessing a 100% success rate. This substantial improvement not only sheds light on the shared susceptibilities across diverse architectures but also demonstrates that attackers can forego the detectable trial-and-error tactics raising increasing the threat of surrogate-based attacks.



## **4. Struggle with Adversarial Defense? Try Diffusion**

cs.CV

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.08273v2) [paper-pdf](http://arxiv.org/pdf/2404.08273v2)

**Authors**: Yujie Li, Yanbin Wang, Haitao Xu, Bin Liu, Jianguo Sun, Zhenhao Guo, Wenrui Ma

**Abstract**: Adversarial attacks induce misclassification by introducing subtle perturbations. Recently, diffusion models are applied to the image classifiers to improve adversarial robustness through adversarial training or by purifying adversarial noise. However, diffusion-based adversarial training often encounters convergence challenges and high computational expenses. Additionally, diffusion-based purification inevitably causes data shift and is deemed susceptible to stronger adaptive attacks. To tackle these issues, we propose the Truth Maximization Diffusion Classifier (TMDC), a generative Bayesian classifier that builds upon pre-trained diffusion models and the Bayesian theorem. Unlike data-driven classifiers, TMDC, guided by Bayesian principles, utilizes the conditional likelihood from diffusion models to determine the class probabilities of input images, thereby insulating against the influences of data shift and the limitations of adversarial training. Moreover, to enhance TMDC's resilience against more potent adversarial attacks, we propose an optimization strategy for diffusion classifiers. This strategy involves post-training the diffusion model on perturbed datasets with ground-truth labels as conditions, guiding the diffusion model to learn the data distribution and maximizing the likelihood under the ground-truth labels. The proposed method achieves state-of-the-art performance on the CIFAR10 dataset against heavy white-box attacks and strong adaptive attacks. Specifically, TMDC achieves robust accuracies of 82.81% against $l_{\infty}$ norm-bounded perturbations and 86.05% against $l_{2}$ norm-bounded perturbations, respectively, with $\epsilon=0.05$.



## **5. Advancing the Robustness of Large Language Models through Self-Denoised Smoothing**

cs.CL

Accepted by NAACL 2024. Jiabao, Bairu, Zhen, Guanhua contributed  equally. This is an updated version of the paper: arXiv:2307.07171

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12274v1) [paper-pdf](http://arxiv.org/pdf/2404.12274v1)

**Authors**: Jiabao Ji, Bairu Hou, Zhen Zhang, Guanhua Zhang, Wenqi Fan, Qing Li, Yang Zhang, Gaowen Liu, Sijia Liu, Shiyu Chang

**Abstract**: Although large language models (LLMs) have achieved significant success, their vulnerability to adversarial perturbations, including recent jailbreak attacks, has raised considerable concerns. However, the increasing size of these models and their limited access make improving their robustness a challenging task. Among various defense strategies, randomized smoothing has shown great potential for LLMs, as it does not require full access to the model's parameters or fine-tuning via adversarial training. However, randomized smoothing involves adding noise to the input before model prediction, and the final model's robustness largely depends on the model's performance on these noise corrupted data. Its effectiveness is often limited by the model's sub-optimal performance on noisy data. To address this issue, we propose to leverage the multitasking nature of LLMs to first denoise the noisy inputs and then to make predictions based on these denoised versions. We call this procedure self-denoised smoothing. Unlike previous denoised smoothing techniques in computer vision, which require training a separate model to enhance the robustness of LLMs, our method offers significantly better efficiency and flexibility. Our experimental results indicate that our method surpasses existing methods in both empirical and certified robustness in defending against adversarial attacks for both downstream tasks and human alignments (i.e., jailbreak attacks). Our code is publicly available at https://github.com/UCSB-NLP-Chang/SelfDenoise



## **6. Efficiently Adversarial Examples Generation for Visual-Language Models under Targeted Transfer Scenarios using Diffusion Models**

cs.CV

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.10335v2) [paper-pdf](http://arxiv.org/pdf/2404.10335v2)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Qing Guo

**Abstract**: Targeted transfer-based attacks involving adversarial examples pose a significant threat to large visual-language models (VLMs). However, the state-of-the-art (SOTA) transfer-based attacks incur high costs due to excessive iteration counts. Furthermore, the generated adversarial examples exhibit pronounced adversarial noise and demonstrate limited efficacy in evading defense methods such as DiffPure. To address these issues, inspired by score matching, we introduce AdvDiffVLM, which utilizes diffusion models to generate natural, unrestricted adversarial examples. Specifically, AdvDiffVLM employs Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring the adversarial examples produced contain natural adversarial semantics and thus possess enhanced transferability. Simultaneously, to enhance the quality of adversarial examples further, we employ the GradCAM-guided Mask method to disperse adversarial semantics throughout the image, rather than concentrating them in a specific area. Experimental results demonstrate that our method achieves a speedup ranging from 10X to 30X compared to existing transfer-based attack methods, while maintaining superior quality of adversarial examples. Additionally, the generated adversarial examples possess strong transferability and exhibit increased robustness against adversarial defense methods. Notably, AdvDiffVLM can successfully attack commercial VLMs, including GPT-4V, in a black-box manner.



## **7. Fortify the Guardian, Not the Treasure: Resilient Adversarial Detectors**

cs.CV

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12120v1) [paper-pdf](http://arxiv.org/pdf/2404.12120v1)

**Authors**: Raz Lapid, Almog Dubin, Moshe Sipper

**Abstract**: This paper presents RADAR-Robust Adversarial Detection via Adversarial Retraining-an approach designed to enhance the robustness of adversarial detectors against adaptive attacks, while maintaining classifier performance. An adaptive attack is one where the attacker is aware of the defenses and adapts their strategy accordingly. Our proposed method leverages adversarial training to reinforce the ability to detect attacks, without compromising clean accuracy. During the training phase, we integrate into the dataset adversarial examples, which were optimized to fool both the classifier and the adversarial detector, enabling the adversarial detector to learn and adapt to potential attack scenarios. Experimental evaluations on the CIFAR-10 and SVHN datasets demonstrate that our proposed algorithm significantly improves a detector's ability to accurately identify adaptive adversarial attacks -- without sacrificing clean accuracy.



## **8. Enhance Robustness of Language Models Against Variation Attack through Graph Integration**

cs.CL

12 pages, 4 figures, accepted by COLING 2024

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12014v1) [paper-pdf](http://arxiv.org/pdf/2404.12014v1)

**Authors**: Zi Xiong, Lizhi Qing, Yangyang Kang, Jiawei Liu, Hongsong Li, Changlong Sun, Xiaozhong Liu, Wei Lu

**Abstract**: The widespread use of pre-trained language models (PLMs) in natural language processing (NLP) has greatly improved performance outcomes. However, these models' vulnerability to adversarial attacks (e.g., camouflaged hints from drug dealers), particularly in the Chinese language with its rich character diversity/variation and complex structures, hatches vital apprehension. In this study, we propose a novel method, CHinese vAriatioN Graph Enhancement (CHANGE), to increase the robustness of PLMs against character variation attacks in Chinese content. CHANGE presents a novel approach for incorporating a Chinese character variation graph into the PLMs. Through designing different supplementary tasks utilizing the graph structure, CHANGE essentially enhances PLMs' interpretation of adversarially manipulated text. Experiments conducted in a multitude of NLP tasks show that CHANGE outperforms current language models in combating against adversarial attacks and serves as a valuable contribution to robust language model research. These findings contribute to the groundwork on robust language models and highlight the substantial potential of graph-guided pre-training strategies for real-world applications.



## **9. Exploring DNN Robustness Against Adversarial Attacks Using Approximate Multipliers**

cs.LG

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11665v1) [paper-pdf](http://arxiv.org/pdf/2404.11665v1)

**Authors**: Mohammad Javad Askarizadeh, Ebrahim Farahmand, Jorge Castro-Godinez, Ali Mahani, Laura Cabrera-Quiros, Carlos Salazar-Garcia

**Abstract**: Deep Neural Networks (DNNs) have advanced in many real-world applications, such as healthcare and autonomous driving. However, their high computational complexity and vulnerability to adversarial attacks are ongoing challenges. In this letter, approximate multipliers are used to explore DNN robustness improvement against adversarial attacks. By uniformly replacing accurate multipliers for state-of-the-art approximate ones in DNN layer models, we explore the DNNs robustness against various adversarial attacks in a feasible time. Results show up to 7% accuracy drop due to approximations when no attack is present while improving robust accuracy up to 10% when attacks applied.



## **10. Towards White Box Deep Learning**

cs.LG

16 pages, 12 figures, independent research, v5 changes: Expanded  Abstract and Related Work section; minor wording improvements

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2403.09863v5) [paper-pdf](http://arxiv.org/pdf/2403.09863v5)

**Authors**: Maciej Satkiewicz

**Abstract**: Deep neural networks learn fragile "shortcut" features, rendering them difficult to interpret (black box) and vulnerable to adversarial attacks. This paper proposes semantic features as a general architectural solution to this problem. The main idea is to make features locality-sensitive in the adequate semantic topology of the domain, thus introducing a strong regularization. The proof of concept network is lightweight, inherently interpretable and achieves almost human-level adversarial test metrics - with no adversarial training! These results and the general nature of the approach warrant further research on semantic features. The code is available at https://github.com/314-Foundation/white-box-nn



## **11. Towards Reliable Empirical Machine Unlearning Evaluation: A Game-Theoretic View**

cs.LG

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11577v1) [paper-pdf](http://arxiv.org/pdf/2404.11577v1)

**Authors**: Yiwen Tu, Pingbang Hu, Jiaqi Ma

**Abstract**: Machine unlearning is the process of updating machine learning models to remove the information of specific training data samples, in order to comply with data protection regulations that allow individuals to request the removal of their personal data. Despite the recent development of numerous unlearning algorithms, reliable evaluation of these algorithms remains an open research question. In this work, we focus on membership inference attack (MIA) based evaluation, one of the most common approaches for evaluating unlearning algorithms, and address various pitfalls of existing evaluation metrics that lack reliability. Specifically, we propose a game-theoretic framework that formalizes the evaluation process as a game between unlearning algorithms and MIA adversaries, measuring the data removal efficacy of unlearning algorithms by the capability of the MIA adversaries. Through careful design of the game, we demonstrate that the natural evaluation metric induced from the game enjoys provable guarantees that the existing evaluation metrics fail to satisfy. Furthermore, we propose a practical and efficient algorithm to estimate the evaluation metric induced from the game, and demonstrate its effectiveness through both theoretical analysis and empirical experiments. This work presents a novel and reliable approach to empirically evaluating unlearning algorithms, paving the way for the development of more effective unlearning techniques.



## **12. GenFighter: A Generative and Evolutive Textual Attack Removal**

cs.LG

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11538v1) [paper-pdf](http://arxiv.org/pdf/2404.11538v1)

**Authors**: Md Athikul Islam, Edoardo Serra, Sushil Jajodia

**Abstract**: Adversarial attacks pose significant challenges to deep neural networks (DNNs) such as Transformer models in natural language processing (NLP). This paper introduces a novel defense strategy, called GenFighter, which enhances adversarial robustness by learning and reasoning on the training classification distribution. GenFighter identifies potentially malicious instances deviating from the distribution, transforms them into semantically equivalent instances aligned with the training data, and employs ensemble techniques for a unified and robust response. By conducting extensive experiments, we show that GenFighter outperforms state-of-the-art defenses in accuracy under attack and attack success rate metrics. Additionally, it requires a high number of queries per attack, making the attack more challenging in real scenarios. The ablation study shows that our approach integrates transfer learning, a generative/evolutive procedure, and an ensemble method, providing an effective defense against NLP adversarial attacks.



## **13. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

cs.CR

arXiv admin note: text overlap with arXiv:2310.07152 by other authors

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11121v1) [paper-pdf](http://arxiv.org/pdf/2404.11121v1)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.



## **14. ToDA: Target-oriented Diffusion Attacker against Recommendation System**

cs.CR

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2401.12578v2) [paper-pdf](http://arxiv.org/pdf/2401.12578v2)

**Authors**: Xiaohao Liu, Zhulin Tao, Ting Jiang, He Chang, Yunshan Ma, Xianglin Huang, Xiang Wang

**Abstract**: Recommendation systems (RS) have become indispensable tools for web services to address information overload, thus enhancing user experiences and bolstering platforms' revenues. However, with their increasing ubiquity, security concerns have also emerged. As the public accessibility of RS, they are susceptible to specific malicious attacks where adversaries can manipulate user profiles, leading to biased recommendations. Recent research often integrates additional modules using generative models to craft these deceptive user profiles, ensuring them are imperceptible while causing the intended harm. Albeit their efficacy, these models face challenges of unstable training and the exploration-exploitation dilemma, which can lead to suboptimal results. In this paper, we pioneer to investigate the potential of diffusion models (DMs), for shilling attacks. Specifically, we propose a novel Target-oriented Diffusion Attack model (ToDA). It incorporates a pre-trained autoencoder that transforms user profiles into a high dimensional space, paired with a Latent Diffusion Attacker (LDA)-the core component of ToDA. LDA introduces noise into the profiles within this latent space, adeptly steering the approximation towards targeted items through cross-attention mechanisms. The global horizon, implemented by a bipartite graph, is involved in LDA and derived from the encoded user profile feature. This makes LDA possible to extend the generation outwards the on-processing user feature itself, and bridges the gap between diffused user features and target item features. Extensive experiments compared to several SOTA baselines demonstrate ToDA's effectiveness. Specific studies exploit the elaborative design of ToDA and underscore the potency of advanced generative models in such contexts.



## **15. Design for Trust utilizing Rareness Reduction**

cs.CR

37th International Conference on VLSI Design, 2024

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2302.08984v2) [paper-pdf](http://arxiv.org/pdf/2302.08984v2)

**Authors**: Aruna Jayasena, Prabhat Mishra

**Abstract**: Increasing design complexity and reduced time-to-market have motivated manufacturers to outsource some parts of the System-on-Chip (SoC) design flow to third-party vendors. This provides an opportunity for attackers to introduce hardware Trojans by constructing stealthy triggers consisting of rare events (e.g., rare signals, states, and transitions). There are promising test generation-based hardware Trojan detection techniques that rely on the activation of rare events. In this paper, we investigate rareness reduction as a design-for-trust solution to make it harder for an adversary to hide Trojans (easier for Trojan detection). Specifically, we analyze different avenues to reduce the potential rare trigger cases, including design diversity and area optimization. While there is a good understanding of the relationship between area, power, energy, and performance, this research provides a better insight into the dependency between area and security. Our experimental evaluation demonstrates that area reduction leads to a reduction in rareness. It also reveals that reducing rareness leads to faster Trojan detection as well as improved coverage by Trojan detection methods.



## **16. Fooling Contrastive Language-Image Pre-trained Models with CLIPMasterPrints**

cs.CV

This work was supported by a research grant (40575) from VILLUM  FONDEN

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2307.03798v3) [paper-pdf](http://arxiv.org/pdf/2307.03798v3)

**Authors**: Matthias Freiberger, Peter Kun, Christian Igel, Anders Sundnes Løvlie, Sebastian Risi

**Abstract**: Models leveraging both visual and textual data such as Contrastive Language-Image Pre-training (CLIP), are the backbone of many recent advances in artificial intelligence. In this work, we show that despite their versatility, such models are vulnerable to what we refer to as fooling master images. Fooling master images are capable of maximizing the confidence score of a CLIP model for a significant number of widely varying prompts, while being either unrecognizable or unrelated to the attacked prompts for humans. The existence of such images is problematic as it could be used by bad actors to maliciously interfere with CLIP-trained image retrieval models in production with comparably small effort as a single image can attack many different prompts. We demonstrate how fooling master images for CLIP (CLIPMasterPrints) can be mined using stochastic gradient descent, projected gradient descent, or blackbox optimization. Contrary to many common adversarial attacks, the blackbox optimization approach allows us to mine CLIPMasterPrints even when the weights of the model are not accessible. We investigate the properties of the mined images, and find that images trained on a small number of image captions generalize to a much larger number of semantically related captions. We evaluate possible mitigation strategies, where we increase the robustness of the model and introduce an approach to automatically detect CLIPMasterPrints to sanitize the input of vulnerable models. Finally, we find that vulnerability to CLIPMasterPrints is related to a modality gap in contrastive pre-trained multi-modal networks. Code available at https://github.com/matfrei/CLIPMasterPrints.



## **17. Self-playing Adversarial Language Game Enhances LLM Reasoning**

cs.CL

Preprint

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.10642v1) [paper-pdf](http://arxiv.org/pdf/2404.10642v1)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du

**Abstract**: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate with respect to a target word only visible to the attacker. The attacker aims to induce the defender to utter the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players should have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by Self-Play in this Adversarial language Game (SPAG). With this goal, we let LLMs act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performance uniformly improves on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLM's reasoning ability. The code is at https://github.com/Linear95/SPAG.



## **18. The Critical Node Game**

math.OC

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2303.05961v2) [paper-pdf](http://arxiv.org/pdf/2303.05961v2)

**Authors**: Gabriele Dragotto, Amine Boukhtouta, Andrea Lodi, Mehdi Taobane

**Abstract**: In this work, we introduce a game-theoretic model that assesses the cyber-security risk of cloud networks and informs security experts on the optimal security strategies. Our approach combines game theory, combinatorial optimization, and cyber-security and aims to minimize the unexpected network disruptions caused by malicious cyber-attacks under uncertainty. Methodologically, we introduce the critical node game, a simultaneous and non-cooperative attacker-defender game where each player solves a combinatorial optimization problem parametrized in the variables of the other player. Each player simultaneously commits to a defensive (or attacking) strategy with limited knowledge about the choices of their adversary. We provide a realistic model for the critical node game and propose an algorithm to compute its stable solutions, i.e., its Nash equilibria. Practically, our approach enables security experts to assess the security posture of the cloud network and dynamically adapt the level of cyber-protection deployed on the network. We provide a detailed analysis of a real-world cloud network and demonstrate the efficacy of our approach through extensive computational tests.



## **19. Minerva: A File-Based Ransomware Detector**

cs.CR

14 pages

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2301.11050v2) [paper-pdf](http://arxiv.org/pdf/2301.11050v2)

**Authors**: Dorjan Hitaj, Giulio Pagnotta, Fabio De Gaspari, Lorenzo De Carli, Luigi V. Mancini

**Abstract**: Ransomware attacks have caused billions of dollars in damages in recent years, and are expected to cause billions more in the future. Consequently, significant effort has been devoted to ransomware detection and mitigation. Behavioral-based ransomware detection approaches have garnered considerable attention recently. These behavioral detectors typically rely on process-based behavioral profiles to identify malicious behaviors. However, with an increasing body of literature highlighting the vulnerability of such approaches to evasion attacks, a comprehensive solution to the ransomware problem remains elusive. This paper presents Minerva, a novel robust approach to ransomware detection. Minerva is engineered to be robust by design against evasion attacks, with architectural and feature selection choices informed by their resilience to adversarial manipulation. We conduct a comprehensive analysis of Minerva across a diverse spectrum of ransomware types, encompassing unseen ransomware as well as variants designed specifically to evade Minerva. Our evaluation showcases the ability of Minerva to accurately identify ransomware, generalize to unseen threats, and withstand evasion attacks. Furthermore, Minerva achieves remarkably low detection times, enabling the adoption of data loss prevention techniques with near-zero overhead.



## **20. Adversarial Identity Injection for Semantic Face Image Synthesis**

cs.CV

Paper accepted at CVPR 2024 Biometrics Workshop

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.10408v1) [paper-pdf](http://arxiv.org/pdf/2404.10408v1)

**Authors**: Giuseppe Tarollo, Tomaso Fontanini, Claudio Ferrari, Guido Borghi, Andrea Prati

**Abstract**: Nowadays, deep learning models have reached incredible performance in the task of image generation. Plenty of literature works address the task of face generation and editing, with human and automatic systems that struggle to distinguish what's real from generated. Whereas most systems reached excellent visual generation quality, they still face difficulties in preserving the identity of the starting input subject. Among all the explored techniques, Semantic Image Synthesis (SIS) methods, whose goal is to generate an image conditioned on a semantic segmentation mask, are the most promising, even though preserving the perceived identity of the input subject is not their main concern. Therefore, in this paper, we investigate the problem of identity preservation in face image generation and present an SIS architecture that exploits a cross-attention mechanism to merge identity, style, and semantic features to generate faces whose identities are as similar as possible to the input ones. Experimental results reveal that the proposed method is not only suitable for preserving the identity but is also effective in the face recognition adversarial attack, i.e. hiding a second identity in the generated faces.



## **21. Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code**

cs.CR

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2401.16820v2) [paper-pdf](http://arxiv.org/pdf/2401.16820v2)

**Authors**: Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, Jiaheng Zhang

**Abstract**: Large Language Models (LLMs) have been widely deployed for their remarkable capability to generate texts resembling human language. However, they could be misused by criminals to create deceptive content, such as fake news and phishing emails, which raises ethical concerns. Watermarking is a key technique to mitigate the misuse of LLMs, which embeds a watermark (e.g., a bit string) into a text generated by a LLM. Consequently, this enables the detection of texts generated by a LLM as well as the tracing of generated texts to a specific user. The major limitation of existing watermark techniques is that they cannot accurately or efficiently extract the watermark from a text, especially when the watermark is a long bit string. This key limitation impedes their deployment for real-world applications, e.g., tracing generated texts to a specific user.   This work introduces a novel watermarking method for LLM-generated text grounded in \textbf{error-correction codes} to address this challenge. We provide strong theoretical analysis, demonstrating that under bounded adversarial word/token edits (insertion, deletion, and substitution), our method can correctly extract watermarks, offering a provable robustness guarantee. This breakthrough is also evidenced by our extensive experimental results. The experiments show that our method substantially outperforms existing baselines in both accuracy and robustness on benchmark datasets. For instance, when embedding a bit string of length 12 into a 200-token generated text, our approach attains an impressive match rate of $98.4\%$, surpassing the performance of Yoo et al. (state-of-the-art baseline) at $85.6\%$. When subjected to a copy-paste attack involving the injection of 50 tokens to generated texts with 200 words, our method maintains a substantial match rate of $90.8\%$, while the match rate of Yoo et al. diminishes to below $65\%$.



## **22. Towards a Novel Perspective on Adversarial Examples Driven by Frequency**

cs.LG

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.10202v1) [paper-pdf](http://arxiv.org/pdf/2404.10202v1)

**Authors**: Zhun Zhang, Yi Zeng, Qihe Liu, Shijie Zhou

**Abstract**: Enhancing our understanding of adversarial examples is crucial for the secure application of machine learning models in real-world scenarios. A prevalent method for analyzing adversarial examples is through a frequency-based approach. However, existing research indicates that attacks designed to exploit low-frequency or high-frequency information can enhance attack performance, leading to an unclear relationship between adversarial perturbations and different frequency components. In this paper, we seek to demystify this relationship by exploring the characteristics of adversarial perturbations within the frequency domain. We employ wavelet packet decomposition for detailed frequency analysis of adversarial examples and conduct statistical examinations across various frequency bands. Intriguingly, our findings indicate that significant adversarial perturbations are present within the high-frequency components of low-frequency bands. Drawing on this insight, we propose a black-box adversarial attack algorithm based on combining different frequency bands. Experiments conducted on multiple datasets and models demonstrate that combining low-frequency bands and high-frequency components of low-frequency bands can significantly enhance attack efficiency. The average attack success rate reaches 99\%, surpassing attacks that utilize a single frequency segment. Additionally, we introduce the normalized disturbance visibility index as a solution to the limitations of $L_2$ norm in assessing continuous and discrete perturbations.



## **23. Ti-Patch: Tiled Physical Adversarial Patch for no-reference video quality metrics**

cs.CV

Accepted to WAIT AINL 2024

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09961v1) [paper-pdf](http://arxiv.org/pdf/2404.09961v1)

**Authors**: Victoria Leonenkova, Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Objective no-reference image- and video-quality metrics are crucial in many computer vision tasks. However, state-of-the-art no-reference metrics have become learning-based and are vulnerable to adversarial attacks. The vulnerability of quality metrics imposes restrictions on using such metrics in quality control systems and comparing objective algorithms. Also, using vulnerable metrics as a loss for deep learning model training can mislead training to worsen visual quality. Because of that, quality metrics testing for vulnerability is a task of current interest. This paper proposes a new method for testing quality metrics vulnerability in the physical space. To our knowledge, quality metrics were not previously tested for vulnerability to this attack; they were only tested in the pixel space. We applied a physical adversarial Ti-Patch (Tiled Patch) attack to quality metrics and did experiments both in pixel and physical space. We also performed experiments on the implementation of physical adversarial wallpaper. The proposed method can be used as additional quality metrics in vulnerability evaluation, complementing traditional subjective comparison and vulnerability tests in the pixel space. We made our code and adversarial videos available on GitHub: https://github.com/leonenkova/Ti-Patch.



## **24. Larger-scale Nakamoto-style Blockchains Don't Necessarily Offer Better Security**

cs.CR

IEEE Symposium on Security and Privacy (IEEE SP), 2024

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09895v1) [paper-pdf](http://arxiv.org/pdf/2404.09895v1)

**Authors**: Jannik Albrecht, Sebastien Andreina, Frederik Armknecht, Ghassan Karame, Giorgia Marson, Julian Willingmann

**Abstract**: Extensive research on Nakamoto-style consensus protocols has shown that network delays degrade the security of these protocols. Established results indicate that, perhaps surprisingly, maximal security is achieved when the network is as small as two nodes due to increased delays in larger networks. This contradicts the very foundation of blockchains, namely that decentralization improves security. In this paper, we take a closer look at how the network scale affects security of Nakamoto-style blockchains. We argue that a crucial aspect has been neglected in existing security models: the larger the network, the harder it is for an attacker to control a significant amount of power. To this end, we introduce a probabilistic corruption model to express the increasing difficulty for an attacker to corrupt resources in larger networks. Based on our model, we analyze the impact of the number of nodes on the (maximum) network delay and the fraction of adversarial power. In particular, we show that (1) increasing the number of nodes eventually violates security, but (2) relying on a small number of nodes does not provide decent security provisions either. We then validate our analysis by means of an empirical evaluation emulating hundreds of thousands of nodes in deployments such as Bitcoin, Monero, Cardano, and Ethereum Classic. Based on our empirical analysis, we concretely analyze the impact of various real-world parameters and configurations on the consistency bounds in existing deployments and on the adversarial power that can be tolerated while providing security. As far as we are aware, this is the first work that analytically and empirically explores the real-world tradeoffs achieved by current popular Nakamoto-style deployments.



## **25. Adversarial Nibbler: An Open Red-Teaming Method for Identifying Diverse Harms in Text-to-Image Generation**

cs.CY

15 pages, 6 figures

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2403.12075v2) [paper-pdf](http://arxiv.org/pdf/2403.12075v2)

**Authors**: Jessica Quaye, Alicia Parrish, Oana Inel, Charvi Rastogi, Hannah Rose Kirk, Minsuk Kahng, Erin van Liemt, Max Bartolo, Jess Tsang, Justin White, Nathan Clement, Rafael Mosquera, Juan Ciro, Vijay Janapa Reddi, Lora Aroyo

**Abstract**: With the rise of text-to-image (T2I) generative AI models reaching wide audiences, it is critical to evaluate model robustness against non-obvious attacks to mitigate the generation of offensive images. By focusing on ``implicitly adversarial'' prompts (those that trigger T2I models to generate unsafe images for non-obvious reasons), we isolate a set of difficult safety issues that human creativity is well-suited to uncover. To this end, we built the Adversarial Nibbler Challenge, a red-teaming methodology for crowdsourcing a diverse set of implicitly adversarial prompts. We have assembled a suite of state-of-the-art T2I models, employed a simple user interface to identify and annotate harms, and engaged diverse populations to capture long-tail safety issues that may be overlooked in standard testing. The challenge is run in consecutive rounds to enable a sustained discovery and analysis of safety pitfalls in T2I models.   In this paper, we present an in-depth account of our methodology, a systematic study of novel attack strategies and discussion of safety failures revealed by challenge participants. We also release a companion visualization tool for easy exploration and derivation of insights from the dataset. The first challenge round resulted in over 10k prompt-image pairs with machine annotations for safety. A subset of 1.5k samples contains rich human annotations of harm types and attack styles. We find that 14% of images that humans consider harmful are mislabeled as ``safe'' by machines. We have identified new attack strategies that highlight the complexity of ensuring T2I model robustness. Our findings emphasize the necessity of continual auditing and adaptation as new vulnerabilities emerge. We are confident that this work will enable proactive, iterative safety assessments and promote responsible development of T2I models.



## **26. VFLGAN: Vertical Federated Learning-based Generative Adversarial Network for Vertically Partitioned Data Publication**

cs.LG

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09722v1) [paper-pdf](http://arxiv.org/pdf/2404.09722v1)

**Authors**: Xun Yuan, Yang Yang, Prosanta Gope, Aryan Pasikhani, Biplab Sikdar

**Abstract**: In the current artificial intelligence (AI) era, the scale and quality of the dataset play a crucial role in training a high-quality AI model. However, good data is not a free lunch and is always hard to access due to privacy regulations like the General Data Protection Regulation (GDPR). A potential solution is to release a synthetic dataset with a similar distribution to that of the private dataset. Nevertheless, in some scenarios, it has been found that the attributes needed to train an AI model belong to different parties, and they cannot share the raw data for synthetic data publication due to privacy regulations. In PETS 2023, Xue et al. proposed the first generative adversary network-based model, VertiGAN, for vertically partitioned data publication. However, after thoroughly investigating, we found that VertiGAN is less effective in preserving the correlation among the attributes of different parties. This article proposes a Vertical Federated Learning-based Generative Adversarial Network, VFLGAN, for vertically partitioned data publication to address the above issues. Our experimental results show that compared with VertiGAN, VFLGAN significantly improves the quality of synthetic data. Taking the MNIST dataset as an example, the quality of the synthetic dataset generated by VFLGAN is 3.2 times better than that generated by VertiGAN w.r.t. the Fr\'echet Distance. We also designed a more efficient and effective Gaussian mechanism for the proposed VFLGAN to provide the synthetic dataset with a differential privacy guarantee. On the other hand, differential privacy only gives the upper bound of the worst-case privacy guarantee. This article also proposes a practical auditing scheme that applies membership inference attacks to estimate privacy leakage through the synthetic dataset.



## **27. A Survey of Neural Network Robustness Assessment in Image Recognition**

cs.CV

Corrected typos and grammatical errors in Section 5

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.08285v2) [paper-pdf](http://arxiv.org/pdf/2404.08285v2)

**Authors**: Jie Wang, Jun Ai, Minyan Lu, Haoran Su, Dan Yu, Yutao Zhang, Junda Zhu, Jingyu Liu

**Abstract**: In recent years, there has been significant attention given to the robustness assessment of neural networks. Robustness plays a critical role in ensuring reliable operation of artificial intelligence (AI) systems in complex and uncertain environments. Deep learning's robustness problem is particularly significant, highlighted by the discovery of adversarial attacks on image classification models. Researchers have dedicated efforts to evaluate robustness in diverse perturbation conditions for image recognition tasks. Robustness assessment encompasses two main techniques: robustness verification/ certification for deliberate adversarial attacks and robustness testing for random data corruptions. In this survey, we present a detailed examination of both adversarial robustness (AR) and corruption robustness (CR) in neural network assessment. Analyzing current research papers and standards, we provide an extensive overview of robustness assessment in image recognition. Three essential aspects are analyzed: concepts, metrics, and assessment methods. We investigate the perturbation metrics and range representations used to measure the degree of perturbations on images, as well as the robustness metrics specifically for the robustness conditions of classification models. The strengths and limitations of the existing methods are also discussed, and some potential directions for future research are provided.



## **28. Less is More: Understanding Word-level Textual Adversarial Attack via n-gram Frequency Descend**

cs.CL

To be published in: 2024 IEEE Conference on Artificial Intelligence  (CAI 2024)

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2302.02568v4) [paper-pdf](http://arxiv.org/pdf/2302.02568v4)

**Authors**: Ning Lu, Shengcai Liu, Zhirui Zhang, Qi Wang, Haifeng Liu, Ke Tang

**Abstract**: Word-level textual adversarial attacks have demonstrated notable efficacy in misleading Natural Language Processing (NLP) models. Despite their success, the underlying reasons for their effectiveness and the fundamental characteristics of adversarial examples (AEs) remain obscure. This work aims to interpret word-level attacks by examining their $n$-gram frequency patterns. Our comprehensive experiments reveal that in approximately 90\% of cases, word-level attacks lead to the generation of examples where the frequency of $n$-grams decreases, a tendency we term as the $n$-gram Frequency Descend ($n$-FD). This finding suggests a straightforward strategy to enhance model robustness: training models using examples with $n$-FD. To examine the feasibility of this strategy, we employed the $n$-gram frequency information, as an alternative to conventional loss gradients, to generate perturbed examples in adversarial training. The experiment results indicate that the frequency-based approach performs comparably with the gradient-based approach in improving model robustness. Our research offers a novel and more intuitive perspective for understanding word-level textual adversarial attacks and proposes a new direction to improve model robustness.



## **29. Black-box Adversarial Transferability: An Empirical Study in Cybersecurity Perspective**

cs.CR

Submitted to Computer & Security (Elsevier)

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.10796v1) [paper-pdf](http://arxiv.org/pdf/2404.10796v1)

**Authors**: Khushnaseeb Roshan, Aasim Zafar

**Abstract**: The rapid advancement of artificial intelligence within the realm of cybersecurity raises significant security concerns. The vulnerability of deep learning models in adversarial attacks is one of the major issues. In adversarial machine learning, malicious users try to fool the deep learning model by inserting adversarial perturbation inputs into the model during its training or testing phase. Subsequently, it reduces the model confidence score and results in incorrect classifications. The novel key contribution of the research is to empirically test the black-box adversarial transferability phenomena in cyber attack detection systems. It indicates that the adversarial perturbation input generated through the surrogate model has a similar impact on the target model in producing the incorrect classification. To empirically validate this phenomenon, surrogate and target models are used. The adversarial perturbation inputs are generated based on the surrogate-model for which the hacker has complete information. Based on these adversarial perturbation inputs, both surrogate and target models are evaluated during the inference phase. We have done extensive experimentation over the CICDDoS-2019 dataset, and the results are classified in terms of various performance metrics like accuracy, precision, recall, and f1-score. The findings indicate that any deep learning model is highly susceptible to adversarial attacks, even if the attacker does not have access to the internal details of the target model. The results also indicate that white-box adversarial attacks have a severe impact compared to black-box adversarial attacks. There is a need to investigate and explore adversarial defence techniques to increase the robustness of the deep learning models against adversarial attacks.



## **30. SpamDam: Towards Privacy-Preserving and Adversary-Resistant SMS Spam Detection**

cs.CR

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09481v1) [paper-pdf](http://arxiv.org/pdf/2404.09481v1)

**Authors**: Yekai Li, Rufan Zhang, Wenxin Rong, Xianghang Mi

**Abstract**: In this study, we introduce SpamDam, a SMS spam detection framework designed to overcome key challenges in detecting and understanding SMS spam, such as the lack of public SMS spam datasets, increasing privacy concerns of collecting SMS data, and the need for adversary-resistant detection models. SpamDam comprises four innovative modules: an SMS spam radar that identifies spam messages from online social networks(OSNs); an SMS spam inspector for statistical analysis; SMS spam detectors(SSDs) that enable both central training and federated learning; and an SSD analyzer that evaluates model resistance against adversaries in realistic scenarios. Leveraging SpamDam, we have compiled over 76K SMS spam messages from Twitter and Weibo between 2018 and 2023, forming the largest dataset of its kind. This dataset has enabled new insights into recent spam campaigns and the training of high-performing binary and multi-label classifiers for spam detection. Furthermore, effectiveness of federated learning has been well demonstrated to enable privacy-preserving SMS spam detection. Additionally, we have rigorously tested the adversarial robustness of SMS spam detection models, introducing the novel reverse backdoor attack, which has shown effectiveness and stealthiness in practical tests.



## **31. Crooked indifferentiability of the Feistel Construction**

cs.CR

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09450v1) [paper-pdf](http://arxiv.org/pdf/2404.09450v1)

**Authors**: Alexander Russell, Qiang Tang, Jiadong Zhu

**Abstract**: The Feistel construction is a fundamental technique for building pseudorandom permutations and block ciphers. This paper shows that a simple adaptation of the construction is resistant, even to algorithm substitution attacks -- that is, adversarial subversion -- of the component round functions. Specifically, we establish that a Feistel-based construction with more than $2000n/\log(1/\epsilon)$ rounds can transform a subverted random function -- which disagrees with the original one at a small fraction (denoted by $\epsilon$) of inputs -- into an object that is \emph{crooked-indifferentiable} from a random permutation, even if the adversary is aware of all the randomness used in the transformation. We also provide a lower bound showing that the construction cannot use fewer than $2n/\log(1/\epsilon)$ rounds to achieve crooked-indifferentiable security.



## **32. Watermark-embedded Adversarial Examples for Copyright Protection against Diffusion Models**

cs.CV

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09401v1) [paper-pdf](http://arxiv.org/pdf/2404.09401v1)

**Authors**: Peifei Zhu, Tsubasa Takahashi, Hirokatsu Kataoka

**Abstract**: Diffusion Models (DMs) have shown remarkable capabilities in various image-generation tasks. However, there are growing concerns that DMs could be used to imitate unauthorized creations and thus raise copyright issues. To address this issue, we propose a novel framework that embeds personal watermarks in the generation of adversarial examples. Such examples can force DMs to generate images with visible watermarks and prevent DMs from imitating unauthorized images. We construct a generator based on conditional adversarial networks and design three losses (adversarial loss, GAN loss, and perturbation loss) to generate adversarial examples that have subtle perturbation but can effectively attack DMs to prevent copyright violations. Training a generator for a personal watermark by our method only requires 5-10 samples within 2-3 minutes, and once the generator is trained, it can generate adversarial examples with that watermark significantly fast (0.2s per image). We conduct extensive experiments in various conditional image-generation scenarios. Compared to existing methods that generate images with chaotic textures, our method adds visible watermarks on the generated images, which is a more straightforward way to indicate copyright violations. We also observe that our adversarial examples exhibit good transferability across unknown generative models. Therefore, this work provides a simple yet powerful way to protect copyright from DM-based imitation.



## **33. Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies**

cs.LG

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2404.09349v1) [paper-pdf](http://arxiv.org/pdf/2404.09349v1)

**Authors**: Brian R. Bartoldson, James Diffenderfer, Konstantinos Parasyris, Bhavya Kailkhura

**Abstract**: This paper revisits the simple, long-studied, yet still unsolved problem of making image classifiers robust to imperceptible perturbations. Taking CIFAR10 as an example, SOTA clean accuracy is about $100$%, but SOTA robustness to $\ell_{\infty}$-norm bounded perturbations barely exceeds $70$%. To understand this gap, we analyze how model size, dataset size, and synthetic data quality affect robustness by developing the first scaling laws for adversarial training. Our scaling laws reveal inefficiencies in prior art and provide actionable feedback to advance the field. For instance, we discovered that SOTA methods diverge notably from compute-optimal setups, using excess compute for their level of robustness. Leveraging a compute-efficient setup, we surpass the prior SOTA with $20$% ($70$%) fewer training (inference) FLOPs. We trained various compute-efficient models, with our best achieving $74$% AutoAttack accuracy ($+3$% gain). However, our scaling laws also predict robustness slowly grows then plateaus at $90$%: dwarfing our new SOTA by scaling is impractical, and perfect robustness is impossible. To better understand this predicted limit, we carry out a small-scale human evaluation on the AutoAttack data that fools our top-performing model. Concerningly, we estimate that human performance also plateaus near $90$%, which we show to be attributable to $\ell_{\infty}$-constrained attacks' generation of invalid images not consistent with their original labels. Having characterized limiting roadblocks, we outline promising paths for future research.



## **34. FaceCat: Enhancing Face Recognition Security with a Unified Generative Model Framework**

cs.CV

Under review

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2404.09193v1) [paper-pdf](http://arxiv.org/pdf/2404.09193v1)

**Authors**: Jiawei Chen, Xiao Yang, Yinpeng Dong, Hang Su, Jianteng Peng, Zhaoxia Yin

**Abstract**: Face anti-spoofing (FAS) and adversarial detection (FAD) have been regarded as critical technologies to ensure the safety of face recognition systems. As a consequence of their limited practicality and generalization, some existing methods aim to devise a framework capable of concurrently detecting both threats to address the challenge. Nevertheless, these methods still encounter challenges of insufficient generalization and suboptimal robustness, potentially owing to the inherent drawback of discriminative models. Motivated by the rich structural and detailed features of face generative models, we propose FaceCat which utilizes the face generative model as a pre-trained model to improve the performance of FAS and FAD. Specifically, FaceCat elaborately designs a hierarchical fusion mechanism to capture rich face semantic features of the generative model. These features then serve as a robust foundation for a lightweight head, designed to execute FAS and FAD tasks simultaneously. As relying solely on single-modality data often leads to suboptimal performance, we further propose a novel text-guided multi-modal alignment strategy that utilizes text prompts to enrich feature representation, thereby enhancing performance. For fair evaluations, we build a comprehensive protocol with a wide range of 28 attack types to benchmark the performance. Extensive experiments validate the effectiveness of FaceCat generalizes significantly better and obtains excellent robustness against input transformations.



## **35. Annealing Self-Distillation Rectification Improves Adversarial Training**

cs.LG

Accepted to ICLR 2024

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2305.12118v2) [paper-pdf](http://arxiv.org/pdf/2305.12118v2)

**Authors**: Yu-Yu Wu, Hung-Jui Wang, Shang-Tse Chen

**Abstract**: In standard adversarial training, models are optimized to fit one-hot labels within allowable adversarial perturbation budgets. However, the ignorance of underlying distribution shifts brought by perturbations causes the problem of robust overfitting. To address this issue and enhance adversarial robustness, we analyze the characteristics of robust models and identify that robust models tend to produce smoother and well-calibrated outputs. Based on the observation, we propose a simple yet effective method, Annealing Self-Distillation Rectification (ADR), which generates soft labels as a better guidance mechanism that accurately reflects the distribution shift under attack during adversarial training. By utilizing ADR, we can obtain rectified distributions that significantly improve model robustness without the need for pre-trained models or extensive extra computation. Moreover, our method facilitates seamless plug-and-play integration with other adversarial training techniques by replacing the hard labels in their objectives. We demonstrate the efficacy of ADR through extensive experiments and strong performances across datasets.



## **36. IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2310.11890v3) [paper-pdf](http://arxiv.org/pdf/2310.11890v3)

**Authors**: Yue Cao, Tianlin Li, Xiaofeng Cao, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: We introduce a novel approach to counter adversarial attacks, namely, image resampling. Image resampling transforms a discrete image into a new one, simulating the process of scene recapturing or rerendering as specified by a geometrical transformation. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. To validate this concept, we present a comprehensive study on leveraging image resampling to defend against adversarial attacks. We have developed basic resampling methods that employ interpolation strategies and coordinate shifting magnitudes. Our analysis reveals that these basic methods can partially mitigate adversarial attacks. However, they come with apparent limitations: the accuracy of clean images noticeably decreases, while the improvement in accuracy on adversarial examples is not substantial. We propose implicit representation-driven image resampling (IRAD) to overcome these limitations. First, we construct an implicit continuous representation that enables us to represent any input image within a continuous coordinate space. Second, we introduce SampleNet, which automatically generates pixel-wise shifts for resampling in response to different inputs. Furthermore, we can extend our approach to the state-of-the-art diffusion-based method, accelerating it with fewer time steps while preserving its defense capability. Extensive experiments demonstrate that our method significantly enhances the adversarial robustness of diverse deep models against various attacks while maintaining high accuracy on clean images.



## **37. Proof-of-Learning with Incentive Security**

cs.CR

22 pages, 6 figures

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.09005v1) [paper-pdf](http://arxiv.org/pdf/2404.09005v1)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.



## **38. Stability and Generalization in Free Adversarial Training**

cs.LG

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.08980v1) [paper-pdf](http://arxiv.org/pdf/2404.08980v1)

**Authors**: Xiwei Cheng, Kexin Fu, Farzan Farnia

**Abstract**: While adversarial training methods have resulted in significant improvements in the deep neural nets' robustness against norm-bounded adversarial perturbations, their generalization performance from training samples to test data has been shown to be considerably worse than standard empirical risk minimization methods. Several recent studies seek to connect the generalization behavior of adversarially trained classifiers to various gradient-based min-max optimization algorithms used for their training. In this work, we study the generalization performance of adversarial training methods using the algorithmic stability framework. Specifically, our goal is to compare the generalization performance of the vanilla adversarial training scheme fully optimizing the perturbations at every iteration vs. the free adversarial training simultaneously optimizing the norm-bounded perturbations and classifier parameters. Our proven generalization bounds indicate that the free adversarial training method could enjoy a lower generalization gap between training and test samples due to the simultaneous nature of its min-max optimization algorithm. We perform several numerical experiments to evaluate the generalization performance of vanilla, fast, and free adversarial training methods. Our empirical findings also show the improved generalization performance of the free adversarial training method and further demonstrate that the better generalization result could translate to greater robustness against black-box attack schemes. The code is available at https://github.com/Xiwei-Cheng/Stability_FreeAT.



## **39. Multimodal Attack Detection for Action Recognition Models**

cs.CR

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.10790v1) [paper-pdf](http://arxiv.org/pdf/2404.10790v1)

**Authors**: Furkan Mumcu, Yasin Yilmaz

**Abstract**: Adversarial machine learning attacks on video action recognition models is a growing research area and many effective attacks were introduced in recent years. These attacks show that action recognition models can be breached in many ways. Hence using these models in practice raises significant security concerns. However, there are very few works which focus on defending against or detecting attacks. In this work, we propose a novel universal detection method which is compatible with any action recognition model. In our extensive experiments, we show that our method consistently detects various attacks against different target models with high true positive rates while satisfying very low false positive rates. Tested against four state-of-the-art attacks targeting four action recognition models, the proposed detector achieves an average AUC of 0.911 over 16 test cases while the best performance achieved by the existing detectors is 0.645 average AUC. This 41.2% improvement is enabled by the robustness of the proposed detector to varying attack methods and target models. The lowest AUC achieved by our detector across the 16 test cases is 0.837 while the competing detector's performance drops as low as 0.211. We also show that the proposed detector is robust to varying attack strengths. In addition, we analyze our method's real-time performance with different hardware setups to demonstrate its potential as a practical defense mechanism.



## **40. MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**

cs.LG

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2402.02263v3) [paper-pdf](http://arxiv.org/pdf/2402.02263v3)

**Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi

**Abstract**: Adversarial robustness often comes at the cost of degraded accuracy, impeding the real-life application of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.



## **41. Adversarial Patterns: Building Robust Android Malware Classifiers**

cs.CR

survey

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2203.02121v2) [paper-pdf](http://arxiv.org/pdf/2203.02121v2)

**Authors**: Dipkamal Bhusal, Nidhi Rastogi

**Abstract**: Machine learning models are increasingly being adopted across various fields, such as medicine, business, autonomous vehicles, and cybersecurity, to analyze vast amounts of data, detect patterns, and make predictions or recommendations. In the field of cybersecurity, these models have made significant improvements in malware detection. However, despite their ability to understand complex patterns from unstructured data, these models are susceptible to adversarial attacks that perform slight modifications in malware samples, leading to misclassification from malignant to benign. Numerous defense approaches have been proposed to either detect such adversarial attacks or improve model robustness. These approaches have resulted in a multitude of attack and defense techniques and the emergence of a field known as `adversarial machine learning.' In this survey paper, we provide a comprehensive review of adversarial machine learning in the context of Android malware classifiers. Android is the most widely used operating system globally and is an easy target for malicious agents. The paper first presents an extensive background on Android malware classifiers, followed by an examination of the latest advancements in adversarial attacks and defenses. Finally, the paper provides guidelines for designing robust malware classifiers and outlines research directions for the future.



## **42. PASA: Attack Agnostic Unsupervised Adversarial Detection using Prediction & Attribution Sensitivity Analysis**

cs.CR

9th IEEE European Symposium on Security and Privacy

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.10789v1) [paper-pdf](http://arxiv.org/pdf/2404.10789v1)

**Authors**: Dipkamal Bhusal, Md Tanvirul Alam, Monish K. Veerabhadran, Michael Clifford, Sara Rampazzi, Nidhi Rastogi

**Abstract**: Deep neural networks for classification are vulnerable to adversarial attacks, where small perturbations to input samples lead to incorrect predictions. This susceptibility, combined with the black-box nature of such networks, limits their adoption in critical applications like autonomous driving. Feature-attribution-based explanation methods provide relevance of input features for model predictions on input samples, thus explaining model decisions. However, we observe that both model predictions and feature attributions for input samples are sensitive to noise. We develop a practical method for this characteristic of model prediction and feature attribution to detect adversarial samples. Our method, PASA, requires the computation of two test statistics using model prediction and feature attribution and can reliably detect adversarial samples using thresholds learned from benign samples. We validate our lightweight approach by evaluating the performance of PASA on varying strengths of FGSM, PGD, BIM, and CW attacks on multiple image and non-image datasets. On average, we outperform state-of-the-art statistical unsupervised adversarial detectors on CIFAR-10 and ImageNet by 14\% and 35\% ROC-AUC scores, respectively. Moreover, our approach demonstrates competitive performance even when an adversary is aware of the defense mechanism.



## **43. JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models**

cs.CR

Submitted to VIS 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08793v1) [paper-pdf](http://arxiv.org/pdf/2404.08793v1)

**Authors**: Yingchaojie Feng, Zhizhang Chen, Zhining Kang, Sijia Wang, Minfeng Zhu, Wei Zhang, Wei Chen

**Abstract**: The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.



## **44. What is the Solution for State-Adversarial Multi-Agent Reinforcement Learning?**

cs.AI

Accepted by Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2212.02705v5) [paper-pdf](http://arxiv.org/pdf/2212.02705v5)

**Authors**: Songyang Han, Sanbao Su, Sihong He, Shuo Han, Haizhao Yang, Shaofeng Zou, Fei Miao

**Abstract**: Various methods for Multi-Agent Reinforcement Learning (MARL) have been developed with the assumption that agents' policies are based on accurate state information. However, policies learned through Deep Reinforcement Learning (DRL) are susceptible to adversarial state perturbation attacks. In this work, we propose a State-Adversarial Markov Game (SAMG) and make the first attempt to investigate different solution concepts of MARL under state uncertainties. Our analysis shows that the commonly used solution concepts of optimal agent policy and robust Nash equilibrium do not always exist in SAMGs. To circumvent this difficulty, we consider a new solution concept called robust agent policy, where agents aim to maximize the worst-case expected state value. We prove the existence of robust agent policy for finite state and finite action SAMGs. Additionally, we propose a Robust Multi-Agent Adversarial Actor-Critic (RMA3C) algorithm to learn robust policies for MARL agents under state uncertainties. Our experiments demonstrate that our algorithm outperforms existing methods when faced with state perturbations and greatly improves the robustness of MARL policies. Our code is public on https://songyanghan.github.io/what_is_solution/.



## **45. Mayhem: Targeted Corruption of Register and Stack Variables**

cs.CR

ACM ASIACCS 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2309.02545v2) [paper-pdf](http://arxiv.org/pdf/2309.02545v2)

**Authors**: Andrew J. Adiletta, M. Caner Tol, Yarkın Doröz, Berk Sunar

**Abstract**: In the past decade, many vulnerabilities were discovered in microarchitectures which yielded attack vectors and motivated the study of countermeasures. Further, architectural and physical imperfections in DRAMs led to the discovery of Rowhammer attacks which give an adversary power to introduce bit flips in a victim's memory space. Numerous studies analyzed Rowhammer and proposed techniques to prevent it altogether or to mitigate its effects.   In this work, we push the boundary and show how Rowhammer can be further exploited to inject faults into stack variables and even register values in a victim's process. We achieve this by targeting the register value that is stored in the process's stack, which subsequently is flushed out into the memory, where it becomes vulnerable to Rowhammer. When the faulty value is restored into the register, it will end up used in subsequent iterations. The register value can be stored in the stack via latent function calls in the source or by actively triggering signal handlers. We demonstrate the power of the findings by applying the techniques to bypass SUDO and SSH authentication. We further outline how MySQL and other cryptographic libraries can be targeted with the new attack vector. There are a number of challenges this work overcomes with extensive experimentation before coming together to yield an end-to-end attack on an OpenSSL digital signature: achieving co-location with stack and register variables, with synchronization provided via a blocking window. We show that stack and registers are no longer safe from the Rowhammer attack.



## **46. On the Robustness of Language Guidance for Low-Level Vision Tasks: Findings from Depth Estimation**

cs.CV

Accepted to CVPR 2024. Project webpage:  https://agneetchatterjee.com/robustness_depth_lang/

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08540v1) [paper-pdf](http://arxiv.org/pdf/2404.08540v1)

**Authors**: Agneet Chatterjee, Tejas Gokhale, Chitta Baral, Yezhou Yang

**Abstract**: Recent advances in monocular depth estimation have been made by incorporating natural language as additional guidance. Although yielding impressive results, the impact of the language prior, particularly in terms of generalization and robustness, remains unexplored. In this paper, we address this gap by quantifying the impact of this prior and introduce methods to benchmark its effectiveness across various settings. We generate "low-level" sentences that convey object-centric, three-dimensional spatial relationships, incorporate them as additional language priors and evaluate their downstream impact on depth estimation. Our key finding is that current language-guided depth estimators perform optimally only with scene-level descriptions and counter-intuitively fare worse with low level descriptions. Despite leveraging additional data, these methods are not robust to directed adversarial attacks and decline in performance with an increase in distribution shift. Finally, to provide a foundation for future research, we identify points of failures and offer insights to better understand these shortcomings. With an increasing number of methods using language for depth estimation, our findings highlight the opportunities and pitfalls that require careful consideration for effective deployment in real-world settings



## **47. VertAttack: Taking advantage of Text Classifiers' horizontal vision**

cs.CL

14 pages, 4 figures, accepted to NAACL 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08538v1) [paper-pdf](http://arxiv.org/pdf/2404.08538v1)

**Authors**: Jonathan Rusert

**Abstract**: Text classification systems have continuously improved in performance over the years. However, nearly all current SOTA classifiers have a similar shortcoming, they process text in a horizontal manner. Vertically written words will not be recognized by a classifier. In contrast, humans are easily able to recognize and read words written both horizontally and vertically. Hence, a human adversary could write problematic words vertically and the meaning would still be preserved to other humans. We simulate such an attack, VertAttack. VertAttack identifies which words a classifier is reliant on and then rewrites those words vertically. We find that VertAttack is able to greatly drop the accuracy of 4 different transformer models on 5 datasets. For example, on the SST2 dataset, VertAttack is able to drop RoBERTa's accuracy from 94 to 13%. Furthermore, since VertAttack does not replace the word, meaning is easily preserved. We verify this via a human study and find that crowdworkers are able to correctly label 77% perturbed texts perturbed, compared to 81% of the original texts. We believe VertAttack offers a look into how humans might circumvent classifiers in the future and thus inspire a look into more robust algorithms.



## **48. Adversarially Robust Spiking Neural Networks Through Conversion**

cs.NE

Transactions on Machine Learning Research (TMLR), 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2311.09266v2) [paper-pdf](http://arxiv.org/pdf/2311.09266v2)

**Authors**: Ozan Özdenizci, Robert Legenstein

**Abstract**: Spiking neural networks (SNNs) provide an energy-efficient alternative to a variety of artificial neural network (ANN) based AI applications. As the progress in neuromorphic computing with SNNs expands their use in applications, the problem of adversarial robustness of SNNs becomes more pronounced. To the contrary of the widely explored end-to-end adversarial training based solutions, we address the limited progress in scalable robust SNN training methods by proposing an adversarially robust ANN-to-SNN conversion algorithm. Our method provides an efficient approach to embrace various computationally demanding robust learning objectives that have been proposed for ANNs. During a post-conversion robust finetuning phase, our method adversarially optimizes both layer-wise firing thresholds and synaptic connectivity weights of the SNN to maintain transferred robustness gains from the pre-trained ANN. We perform experimental evaluations in a novel setting proposed to rigorously assess the robustness of SNNs, where numerous adaptive adversarial attacks that account for the spike-based operation dynamics are considered. Results show that our approach yields a scalable state-of-the-art solution for adversarially robust deep SNNs with low-latency.



## **49. Counterfactual Explanations for Face Forgery Detection via Adversarial Removal of Artifacts**

cs.CV

Accepted to ICME2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08341v1) [paper-pdf](http://arxiv.org/pdf/2404.08341v1)

**Authors**: Yang Li, Songlin Yang, Wei Wang, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Highly realistic AI generated face forgeries known as deepfakes have raised serious social concerns. Although DNN-based face forgery detection models have achieved good performance, they are vulnerable to latest generative methods that have less forgery traces and adversarial attacks. This limitation of generalization and robustness hinders the credibility of detection results and requires more explanations. In this work, we provide counterfactual explanations for face forgery detection from an artifact removal perspective. Specifically, we first invert the forgery images into the StyleGAN latent space, and then adversarially optimize their latent representations with the discrimination supervision from the target detection model. We verify the effectiveness of the proposed explanations from two aspects: (1) Counterfactual Trace Visualization: the enhanced forgery images are useful to reveal artifacts by visually contrasting the original images and two different visualization methods; (2) Transferable Adversarial Attacks: the adversarial forgery images generated by attacking the detection model are able to mislead other detection models, implying the removed artifacts are general. Extensive experiments demonstrate that our method achieves over 90% attack success rate and superior attack transferability. Compared with naive adversarial noise methods, our method adopts both generative and discriminative model priors, and optimize the latent representations in a synthesis-by-analysis way, which forces the search of counterfactual explanations on the natural face manifold. Thus, more general counterfactual traces can be found and better adversarial attack transferability can be achieved.



## **50. Combating Advanced Persistent Threats: Challenges and Solutions**

cs.CR

This work has been accepted by IEEE NETWORK in April 2024. 9 pages, 5  figures, 1 table

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2309.09498v2) [paper-pdf](http://arxiv.org/pdf/2309.09498v2)

**Authors**: Yuntao Wang, Han Liu, Zhendong Li, Zhou Su, Jiliang Li

**Abstract**: The rise of advanced persistent threats (APTs) has marked a significant cybersecurity challenge, characterized by sophisticated orchestration, stealthy execution, extended persistence, and targeting valuable assets across diverse sectors. Provenance graph-based kernel-level auditing has emerged as a promising approach to enhance visibility and traceability within intricate network environments. However, it still faces challenges including reconstructing complex lateral attack chains, detecting dynamic evasion behaviors, and defending smart adversarial subgraphs. To bridge the research gap, this paper proposes an efficient and robust APT defense scheme leveraging provenance graphs, including a network-level distributed audit model for cost-effective lateral attack reconstruction, a trust-oriented APT evasion behavior detection strategy, and a hidden Markov model based adversarial subgraph defense approach. Through prototype implementation and extensive experiments, we validate the effectiveness of our system. Lastly, crucial open research directions are outlined in this emerging field.



