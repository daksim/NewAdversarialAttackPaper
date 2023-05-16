# Latest Adversarial Attack Papers
**update at 2023-05-16 09:53:55**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Attacking Perceptual Similarity Metrics**

cs.CV

TMLR 2023 (Featured Certification). Code is available at  https://tinyurl.com/attackingpsm

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08840v1) [paper-pdf](http://arxiv.org/pdf/2305.08840v1)

**Authors**: Abhijay Ghildyal, Feng Liu

**Abstract**: Perceptual similarity metrics have progressively become more correlated with human judgments on perceptual similarity; however, despite recent advances, the addition of an imperceptible distortion can still compromise these metrics. In our study, we systematically examine the robustness of these metrics to imperceptible adversarial perturbations. Following the two-alternative forced-choice experimental design with two distorted images and one reference image, we perturb the distorted image closer to the reference via an adversarial attack until the metric flips its judgment. We first show that all metrics in our study are susceptible to perturbations generated via common adversarial attacks such as FGSM, PGD, and the One-pixel attack. Next, we attack the widely adopted LPIPS metric using spatial-transformation-based adversarial perturbations (stAdv) in a white-box setting to craft adversarial examples that can effectively transfer to other similarity metrics in a black-box setting. We also combine the spatial attack stAdv with PGD ($\ell_\infty$-bounded) attack to increase transferability and use these adversarial examples to benchmark the robustness of both traditional and recently developed metrics. Our benchmark provides a good starting point for discussion and further research on the robustness of metrics to imperceptible adversarial perturbations.



## **2. Defending Against Misinformation Attacks in Open-Domain Question Answering**

cs.CL

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2212.10002v2) [paper-pdf](http://arxiv.org/pdf/2212.10002v2)

**Authors**: Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, Benjamin Van Durme

**Abstract**: Recent work in open-domain question answering (ODQA) has shown that adversarial poisoning of the search collection can cause large drops in accuracy for production systems. However, little to no work has proposed methods to defend against these attacks. To do so, we rely on the intuition that redundant information often exists in large corpora. To find it, we introduce a method that uses query augmentation to search for a diverse set of passages that could answer the original question but are less likely to have been poisoned. We integrate these new passages into the model through the design of a novel confidence method, comparing the predicted answer to its appearance in the retrieved contexts (what we call \textit{Confidence from Answer Redundancy}, i.e. CAR). Together these methods allow for a simple but effective way to defend against poisoning attacks that provides gains of nearly 20\% exact match across varying levels of data poisoning/knowledge conflicts.



## **3. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

cs.CV

Code Page: https://github.com/WindVChen/DiffAttack

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08192v1) [paper-pdf](http://arxiv.org/pdf/2305.08192v1)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an additional recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into adversarial attack field. Extensive experiments on various model structures (including CNNs, Transformers, MLPs) and defense methods have demonstrated our superiority over other attack methods.



## **4. Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures**

cs.IR

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08183v1) [paper-pdf](http://arxiv.org/pdf/2305.08183v1)

**Authors**: Wei Yuan, Shilong Yuan, Kai Zheng, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated recommender systems (FedRecs) have been widely explored recently due to their ability to protect user data privacy. In FedRecs, a central server collaboratively learns recommendation models by sharing model public parameters with clients, thereby offering a privacy-preserving solution. Unfortunately, the exposure of model parameters leaves a backdoor for adversaries to manipulate FedRecs. Existing works about FedRec security already reveal that items can easily be promoted by malicious users via model poisoning attacks, but all of them mainly focus on FedRecs with only collaborative information (i.e., user-item interactions). We argue that these attacks are effective because of the data sparsity of collaborative signals. In practice, auxiliary information, such as products' visual descriptions, is used to alleviate collaborative filtering data's sparsity. Therefore, when incorporating visual information in FedRecs, all existing model poisoning attacks' effectiveness becomes questionable. In this paper, we conduct extensive experiments to verify that incorporating visual information can beat existing state-of-the-art attacks in reasonable settings. However, since visual information is usually provided by external sources, simply including it will create new security problems. Specifically, we propose a new kind of poisoning attack for visually-aware FedRecs, namely image poisoning attacks, where adversaries can gradually modify the uploaded image to manipulate item ranks during FedRecs' training process. Furthermore, we reveal that the potential collaboration between image poisoning attacks and model poisoning attacks will make visually-aware FedRecs more vulnerable to being manipulated. To safely use visual information, we employ a diffusion model in visually-aware FedRecs to purify each uploaded image and detect the adversarial images.



## **5. Improving Defensive Distillation using Teacher Assistant**

cs.CV

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08076v1) [paper-pdf](http://arxiv.org/pdf/2305.08076v1)

**Authors**: Maniratnam Mandal, Suna Gao

**Abstract**: Adversarial attacks pose a significant threat to the security and safety of deep neural networks being applied to modern applications. More specifically, in computer vision-based tasks, experts can use the knowledge of model architecture to create adversarial samples imperceptible to the human eye. These attacks can lead to security problems in popular applications such as self-driving cars, face recognition, etc. Hence, building networks which are robust to such attacks is highly desirable and essential. Among the various methods present in literature, defensive distillation has shown promise in recent years. Using knowledge distillation, researchers have been able to create models robust against some of those attacks. However, more attacks have been developed exposing weakness in defensive distillation. In this project, we derive inspiration from teacher assistant knowledge distillation and propose that introducing an assistant network can improve the robustness of the distilled model. Through a series of experiments, we evaluate the distilled models for different distillation temperatures in terms of accuracy, sensitivity, and robustness. Our experiments demonstrate that the proposed hypothesis can improve robustness in most cases. Additionally, we show that multi-step distillation can further improve robustness with very little impact on model accuracy.



## **6. DNN-Defender: An in-DRAM Deep Neural Network Defense Mechanism for Adversarial Weight Attack**

cs.CR

10 pages, 11 figures

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08034v1) [paper-pdf](http://arxiv.org/pdf/2305.08034v1)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: With deep learning deployed in many security-sensitive areas, machine learning security is becoming progressively important. Recent studies demonstrate attackers can exploit system-level techniques exploiting the RowHammer vulnerability of DRAM to deterministically and precisely flip bits in Deep Neural Networks (DNN) model weights to affect inference accuracy. The existing defense mechanisms are software-based, such as weight reconstruction requiring expensive training overhead or performance degradation. On the other hand, generic hardware-based victim-/aggressor-focused mechanisms impose expensive hardware overheads and preserve the spatial connection between victim and aggressor rows. In this paper, we present the first DRAM-based victim-focused defense mechanism tailored for quantized DNNs, named DNN-Defender that leverages the potential of in-DRAM swapping to withstand the targeted bit-flip attacks. Our results indicate that DNN-Defender can deliver a high level of protection downgrading the performance of targeted RowHammer attacks to a random attack level. In addition, the proposed defense has no accuracy drop on CIFAR-10 and ImageNet datasets without requiring any software training or incurring additional hardware overhead.



## **7. On enhancing the robustness of Vision Transformers: Defensive Diffusion**

cs.CV

Our code is publicly available at  https://github.com/Muhammad-Huzaifaa/Defensive_Diffusion

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08031v1) [paper-pdf](http://arxiv.org/pdf/2305.08031v1)

**Authors**: Raza Imam, Muhammad Huzaifa, Mohammed El-Amine Azz

**Abstract**: Privacy and confidentiality of medical data are of utmost importance in healthcare settings. ViTs, the SOTA vision model, rely on large amounts of patient data for training, which raises concerns about data security and the potential for unauthorized access. Adversaries may exploit vulnerabilities in ViTs to extract sensitive patient information and compromising patient privacy. This work address these vulnerabilities to ensure the trustworthiness and reliability of ViTs in medical applications. In this work, we introduced a defensive diffusion technique as an adversarial purifier to eliminate adversarial noise introduced by attackers in the original image. By utilizing the denoising capabilities of the diffusion model, we employ a reverse diffusion process to effectively eliminate the adversarial noise from the attack sample, resulting in a cleaner image that is then fed into the ViT blocks. Our findings demonstrate the effectiveness of the diffusion model in eliminating attack-agnostic adversarial noise from images. Additionally, we propose combining knowledge distillation with our framework to obtain a lightweight student model that is both computationally efficient and robust against gray box attacks. Comparison of our method with a SOTA baseline method, SEViT, shows that our work is able to outperform the baseline. Extensive experiments conducted on a publicly available Tuberculosis X-ray dataset validate the computational efficiency and improved robustness achieved by our proposed architecture.



## **8. Detection and Mitigation of Byzantine Attacks in Distributed Training**

cs.LG

21 pages, 17 figures, 6 tables. The material in this work appeared in  part at arXiv:2108.02416 which has been published at the 2022 IEEE  International Symposium on Information Theory

**SubmitDate**: 2023-05-13    [abs](http://arxiv.org/abs/2208.08085v4) [paper-pdf](http://arxiv.org/pdf/2208.08085v4)

**Authors**: Konstantinos Konstantinidis, Namrata Vaswani, Aditya Ramamoorthy

**Abstract**: A plethora of modern machine learning tasks require the utilization of large-scale distributed clusters as a critical component of the training pipeline. However, abnormal Byzantine behavior of the worker nodes can derail the training and compromise the quality of the inference. Such behavior can be attributed to unintentional system malfunctions or orchestrated attacks; as a result, some nodes may return arbitrary results to the parameter server (PS) that coordinates the training. Recent work considers a wide range of attack models and has explored robust aggregation and/or computational redundancy to correct the distorted gradients.   In this work, we consider attack models ranging from strong ones: $q$ omniscient adversaries with full knowledge of the defense protocol that can change from iteration to iteration to weak ones: $q$ randomly chosen adversaries with limited collusion abilities which only change every few iterations at a time. Our algorithms rely on redundant task assignments coupled with detection of adversarial behavior. We also show the convergence of our method to the optimal point under common assumptions and settings considered in literature. For strong attacks, we demonstrate a reduction in the fraction of distorted gradients ranging from 16%-99% as compared to the prior state-of-the-art. Our top-1 classification accuracy results on the CIFAR-10 data set demonstrate 25% advantage in accuracy (averaged over strong and weak scenarios) under the most sophisticated attacks compared to state-of-the-art methods.



## **9. Quantum Lock: A Provable Quantum Communication Advantage**

quant-ph

47 pages, 13 figures

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2110.09469v4) [paper-pdf](http://arxiv.org/pdf/2110.09469v4)

**Authors**: Kaushik Chakraborty, Mina Doosti, Yao Ma, Chirag Wadhwa, Myrto Arapinis, Elham Kashefi

**Abstract**: Physical unclonable functions(PUFs) provide a unique fingerprint to a physical entity by exploiting the inherent physical randomness. Gao et al. discussed the vulnerability of most current-day PUFs to sophisticated machine learning-based attacks. We address this problem by integrating classical PUFs and existing quantum communication technology. Specifically, this paper proposes a generic design of provably secure PUFs, called hybrid locked PUFs(HLPUFs), providing a practical solution for securing classical PUFs. An HLPUF uses a classical PUF(CPUF), and encodes the output into non-orthogonal quantum states to hide the outcomes of the underlying CPUF from any adversary. Here we introduce a quantum lock to protect the HLPUFs from any general adversaries. The indistinguishability property of the non-orthogonal quantum states, together with the quantum lockdown technique prevents the adversary from accessing the outcome of the CPUFs. Moreover, we show that by exploiting non-classical properties of quantum states, the HLPUF allows the server to reuse the challenge-response pairs for further client authentication. This result provides an efficient solution for running PUF-based client authentication for an extended period while maintaining a small-sized challenge-response pairs database on the server side. Later, we support our theoretical contributions by instantiating the HLPUFs design using accessible real-world CPUFs. We use the optimal classical machine-learning attacks to forge both the CPUFs and HLPUFs, and we certify the security gap in our numerical simulation for construction which is ready for implementation.



## **10. Two-in-One: A Model Hijacking Attack Against Text Generation Models**

cs.CR

To appear in the 32nd USENIX Security Symposium, August 2023,  Anaheim, CA, USA

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07406v1) [paper-pdf](http://arxiv.org/pdf/2305.07406v1)

**Authors**: Wai Man Si, Michael Backes, Yang Zhang, Ahmed Salem

**Abstract**: Machine learning has progressed significantly in various applications ranging from face recognition to text generation. However, its success has been accompanied by different attacks. Recently a new attack has been proposed which raises both accountability and parasitic computing risks, namely the model hijacking attack. Nevertheless, this attack has only focused on image classification tasks. In this work, we broaden the scope of this attack to include text generation and classification models, hence showing its broader applicability. More concretely, we propose a new model hijacking attack, Ditto, that can hijack different text classification tasks into multiple generation ones, e.g., language translation, text summarization, and language modeling. We use a range of text benchmark datasets such as SST-2, TweetEval, AGnews, QNLI, and IMDB to evaluate the performance of our attacks. Our results show that by using Ditto, an adversary can successfully hijack text generation models without jeopardizing their utility.



## **11. Novel bribery mining attacks in the bitcoin system and the bribery miner's dilemma**

cs.GT

26 pages, 16 figures, 3 tables

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07381v1) [paper-pdf](http://arxiv.org/pdf/2305.07381v1)

**Authors**: Junjie Hu, Chunxiang Xu, Zhe Jiang, Jiwu Cao

**Abstract**: Mining attacks allow adversaries to obtain a disproportionate share of the mining reward by deviating from the honest mining strategy in the Bitcoin system. Among them, the most well-known are selfish mining (SM), block withholding (BWH), fork after withholding (FAW) and bribery mining. In this paper, we propose two novel mining attacks: bribery semi-selfish mining (BSSM) and bribery stubborn mining (BSM). Both of them can increase the relative extra reward of the adversary and will make the target bribery miners suffer from the bribery miner dilemma. All targets earn less under the Nash equilibrium. For each target, their local optimal strategy is to accept the bribes. However, they will suffer losses, comparing with denying the bribes. Furthermore, for all targets, their global optimal strategy is to deny the bribes. Quantitative analysis and simulation have been verified our theoretical analysis. We propose practical measures to mitigate more advanced mining attack strategies based on bribery mining, and provide new ideas for addressing bribery mining attacks in the future. However, how to completely and effectively prevent these attacks is still needed on further research.



## **12. Efficient Search of Comprehensively Robust Neural Architectures via Multi-fidelity Evaluation**

cs.CV

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07308v1) [paper-pdf](http://arxiv.org/pdf/2305.07308v1)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstract**: Neural architecture search (NAS) has emerged as one successful technique to find robust deep neural network (DNN) architectures. However, most existing robustness evaluations in NAS only consider $l_{\infty}$ norm-based adversarial noises. In order to improve the robustness of DNN models against multiple types of noises, it is necessary to consider a comprehensive evaluation in NAS for robust architectures. But with the increasing number of types of robustness evaluations, it also becomes more time-consuming to find comprehensively robust architectures. To alleviate this problem, we propose a novel efficient search of comprehensively robust neural architectures via multi-fidelity evaluation (ES-CRNA-ME). Specifically, we first search for comprehensively robust architectures under multiple types of evaluations using the weight-sharing-based NAS method, including different $l_{p}$ norm attacks, semantic adversarial attacks, and composite adversarial attacks. In addition, we reduce the number of robustness evaluations by the correlation analysis, which can incorporate similar evaluations and decrease the evaluation cost. Finally, we propose a multi-fidelity online surrogate during optimization to further decrease the search cost. On the basis of the surrogate constructed by low-fidelity data, the online high-fidelity data is utilized to finetune the surrogate. Experiments on CIFAR10 and CIFAR100 datasets show the effectiveness of our proposed method.



## **13. Parameter identifiability of a deep feedforward ReLU neural network**

math.ST

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2112.12982v2) [paper-pdf](http://arxiv.org/pdf/2112.12982v2)

**Authors**: Joachim Bona-Pellissier, François Bachoc, François Malgouyres

**Abstract**: The possibility for one to recover the parameters-weights and biases-of a neural network thanks to the knowledge of its function on a subset of the input space can be, depending on the situation, a curse or a blessing. On one hand, recovering the parameters allows for better adversarial attacks and could also disclose sensitive information from the dataset used to construct the network. On the other hand, if the parameters of a network can be recovered, it guarantees the user that the features in the latent spaces can be interpreted. It also provides foundations to obtain formal guarantees on the performances of the network. It is therefore important to characterize the networks whose parameters can be identified and those whose parameters cannot. In this article, we provide a set of conditions on a deep fully-connected feedforward ReLU neural network under which the parameters of the network are uniquely identified-modulo permutation and positive rescaling-from the function it implements on a subset of the input space.



## **14. Physical-layer Adversarial Robustness for Deep Learning-based Semantic Communications**

eess.SP

17 pages, 28 figures, accepted by IEEE jsac

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07220v1) [paper-pdf](http://arxiv.org/pdf/2305.07220v1)

**Authors**: Guoshun Nan, Zhichun Li, Jinli Zhai, Qimei Cui, Gong Chen, Xin Du, Xuefei Zhang, Xiaofeng Tao, Zhu Han, Tony Q. S. Quek

**Abstract**: End-to-end semantic communications (ESC) rely on deep neural networks (DNN) to boost communication efficiency by only transmitting the semantics of data, showing great potential for high-demand mobile applications. We argue that central to the success of ESC is the robust interpretation of conveyed semantics at the receiver side, especially for security-critical applications such as automatic driving and smart healthcare. However, robustifying semantic interpretation is challenging as ESC is extremely vulnerable to physical-layer adversarial attacks due to the openness of wireless channels and the fragileness of neural models. Toward ESC robustness in practice, we ask the following two questions: Q1: For attacks, is it possible to generate semantic-oriented physical-layer adversarial attacks that are imperceptible, input-agnostic and controllable? Q2: Can we develop a defense strategy against such semantic distortions and previously proposed adversaries? To this end, we first present MobileSC, a novel semantic communication framework that considers the computation and memory efficiency in wireless environments. Equipped with this framework, we propose SemAdv, a physical-layer adversarial perturbation generator that aims to craft semantic adversaries over the air with the abovementioned criteria, thus answering the Q1. To better characterize the realworld effects for robust training and evaluation, we further introduce a novel adversarial training method SemMixed to harden the ESC against SemAdv attacks and existing strong threats, thus answering the Q2. Extensive experiments on three public benchmarks verify the effectiveness of our proposed methods against various physical adversarial attacks. We also show some interesting findings, e.g., our MobileSC can even be more robust than classical block-wise communication systems in the low SNR regime.



## **15. Stratified Adversarial Robustness with Rejection**

cs.LG

Paper published at International Conference on Machine Learning  (ICML'23)

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.01139v2) [paper-pdf](http://arxiv.org/pdf/2305.01139v2)

**Authors**: Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha

**Abstract**: Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method -- Adversarial Training with Consistent Prediction-based Rejection (CPR) -- for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.



## **16. A theoretical basis for Blockchain Extractable Value**

cs.CR

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2302.02154v2) [paper-pdf](http://arxiv.org/pdf/2302.02154v2)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Extractable Value refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream protocols, like e.g. decentralized exchanges, are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the growing impact of these attacks in the real world, theoretical foundations are still missing. We propose a formal theory of Extractable Value, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against Extractable Value attacks.



## **17. Improving Hyperspectral Adversarial Robustness Under Multiple Attacks**

cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2210.16346v4) [paper-pdf](http://arxiv.org/pdf/2210.16346v4)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation models classifying hyperspectral images (HSI) are vulnerable to adversarial examples. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease in performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network.



## **18. Run-Off Election: Improved Provable Defense against Data Poisoning Attacks**

cs.LG

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2302.02300v2) [paper-pdf](http://arxiv.org/pdf/2302.02300v2)

**Authors**: Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi

**Abstract**: In data poisoning attacks, an adversary tries to change a model's prediction by adding, modifying, or removing samples in the training data. Recently, ensemble-based approaches for obtaining provable defenses against data poisoning have been proposed where predictions are done by taking a majority vote across multiple base models. In this work, we show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. Based on this approach, we propose DPA+ROE and FA+ROE defense methods based on Deep Partition Aggregation (DPA) and Finite Aggregation (FA) approaches from prior work. We evaluate our methods on MNIST, CIFAR-10, and GTSRB and obtain improvements in certified accuracy by up to 3%-4%. Also, by applying ROE on a boosted version of DPA, we gain improvements around 12%-27% comparing to the current state-of-the-art, establishing a new state-of-the-art in (pointwise) certified robustness against data poisoning. In many cases, our approach outperforms the state-of-the-art, even when using 32 times less computational power.



## **19. Untargeted Near-collision Attacks in Biometric Recognition**

cs.CR

Addition of results and correction of typos

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2304.01580v2) [paper-pdf](http://arxiv.org/pdf/2304.01580v2)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Kevin Thiry-Atighehchi

**Abstract**: A biometric recognition system can operate in two distinct modes, identification or verification. In the first mode, the system recognizes an individual by searching the enrolled templates of all the users for a match. In the second mode, the system validates a user's identity claim by comparing the fresh provided template with the enrolled template. The biometric transformation schemes usually produce binary templates that are better handled by cryptographic schemes, and the comparison is based on a distance that leaks information about the similarities between two biometric templates. Both the experimentally determined false match rate and false non-match rate through recognition threshold adjustment define the recognition accuracy, and hence the security of the system. To the best of our knowledge, few works provide a formal treatment of the security under minimum leakage of information, i.e., the binary outcome of a comparison with a threshold. In this paper, we rely on probabilistic modelling to quantify the security strength of binary templates. We investigate the influence of template size, database size and threshold on the probability of having a near-collision. We highlight several untargeted attacks on biometric systems considering naive and adaptive adversaries. Interestingly, these attacks can be launched both online and offline and, both in the identification mode and in the verification mode. We discuss the choice of parameters through the generic presented attacks.



## **20. Distracting Downpour: Adversarial Weather Attacks for Motion Estimation**

cs.CV

This work is a direct extension of our extended abstract from  arXiv:2210.11242

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06716v1) [paper-pdf](http://arxiv.org/pdf/2305.06716v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks on motion estimation, or optical flow, optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, adverse weather conditions constitute a much more realistic threat scenario. Hence, in this work, we present a novel attack on motion estimation that exploits adversarially optimized particles to mimic weather effects like snowflakes, rain streaks or fog clouds. At the core of our attack framework is a differentiable particle rendering system that integrates particles (i) consistently over multiple time steps (ii) into the 3D space (iii) with a photo-realistic appearance. Through optimization, we obtain adversarial weather that significantly impacts the motion estimation. Surprisingly, methods that previously showed good robustness towards small per-pixel perturbations are particularly vulnerable to adversarial weather. At the same time, augmenting the training with non-optimized weather increases a method's robustness towards weather effects and improves generalizability at almost no additional cost.



## **21. Beyond the Model: Data Pre-processing Attack to Deep Learning Models in Android Apps**

cs.CR

Accepted to AsiaCCS WorkShop on Secure and Trustworthy Deep Learning  Systems (SecTL 2023)

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.03963v2) [paper-pdf](http://arxiv.org/pdf/2305.03963v2)

**Authors**: Ye Sang, Yujin Huang, Shuo Huang, Helei Cui

**Abstract**: The increasing popularity of deep learning (DL) models and the advantages of computing, including low latency and bandwidth savings on smartphones, have led to the emergence of intelligent mobile applications, also known as DL apps, in recent years. However, this technological development has also given rise to several security concerns, including adversarial examples, model stealing, and data poisoning issues. Existing works on attacks and countermeasures for on-device DL models have primarily focused on the models themselves. However, scant attention has been paid to the impact of data processing disturbance on the model inference. This knowledge disparity highlights the need for additional research to fully comprehend and address security issues related to data processing for on-device models. In this paper, we introduce a data processing-based attacks against real-world DL apps. In particular, our attack could influence the performance and latency of the model without affecting the operation of a DL app. To demonstrate the effectiveness of our attack, we carry out an empirical study on 517 real-world DL apps collected from Google Play. Among 320 apps utilizing MLkit, we find that 81.56\% of them can be successfully attacked.   The results emphasize the importance of DL app developers being aware of and taking actions to secure on-device models from the perspective of data processing.



## **22. On the Robustness of Graph Neural Diffusion to Topology Perturbations**

cs.LG

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2209.07754v2) [paper-pdf](http://arxiv.org/pdf/2209.07754v2)

**Authors**: Yang Song, Qiyu Kang, Sijie Wang, Zhao Kai, Wee Peng Tay

**Abstract**: Neural diffusion on graphs is a novel class of graph neural networks that has attracted increasing attention recently. The capability of graph neural partial differential equations (PDEs) in addressing common hurdles of graph neural networks (GNNs), such as the problems of over-smoothing and bottlenecks, has been investigated but not their robustness to adversarial attacks. In this work, we explore the robustness properties of graph neural PDEs. We empirically demonstrate that graph neural PDEs are intrinsically more robust against topology perturbation as compared to other GNNs. We provide insights into this phenomenon by exploiting the stability of the heat semigroup under graph topology perturbations. We discuss various graph diffusion operators and relate them to existing graph neural PDEs. Furthermore, we propose a general graph neural PDE framework based on which a new class of robust GNNs can be defined. We verify that the new model achieves comparable state-of-the-art performance on several benchmark datasets.



## **23. Prevention of shoulder-surfing attacks using shifting condition using digraph substitution rules**

cs.CR

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06549v1) [paper-pdf](http://arxiv.org/pdf/2305.06549v1)

**Authors**: Amanul Islam, Fazidah Othman, Nazmus Sakib, Hafiz Md. Hasan Babu

**Abstract**: Graphical passwords are implemented as an alternative scheme to replace alphanumeric passwords to help users to memorize their password. However, most of the graphical password systems are vulnerable to shoulder-surfing attack due to the usage of the visual interface. In this research, a method that uses shifting condition with digraph substitution rules is proposed to address shoulder-surfing attack problem. The proposed algorithm uses both password images and decoy images throughout the user authentication procedure to confuse adversaries from obtaining the password images via direct observation or watching from a recorded session. The pass-images generated by this suggested algorithm are random and can only be generated if the algorithm is fully understood. As a result, adversaries will have no clue to obtain the right password images to log in. A user study was undertaken to assess the proposed method's effectiveness to avoid shoulder-surfing attacks. The results of the user study indicate that the proposed approach can withstand shoulder-surfing attacks (both direct observation and video recording method).The proposed method was tested and the results showed that it is able to resist shoulder-surfing and frequency of occurrence analysis attacks. Moreover, the experience gained in this research can be pervaded the gap on the realm of knowledge of the graphical password.



## **24. Inter-frame Accelerate Attack against Video Interpolation Models**

cs.CV

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06540v1) [paper-pdf](http://arxiv.org/pdf/2305.06540v1)

**Authors**: Junpei Liao, Zhikai Chen, Liang Yi, Wenyuan Yang, Baoyuan Wu, Xiaochun Cao

**Abstract**: Deep learning based video frame interpolation (VIF) method, aiming to synthesis the intermediate frames to enhance video quality, have been highly developed in the past few years. This paper investigates the adversarial robustness of VIF models. We apply adversarial attacks to VIF models and find that the VIF models are very vulnerable to adversarial examples. To improve attack efficiency, we suggest to make full use of the property of video frame interpolation task. The intuition is that the gap between adjacent frames would be small, leading to the corresponding adversarial perturbations being similar as well. Then we propose a novel attack method named Inter-frame Accelerate Attack (IAA) that initializes the perturbation as the perturbation for the previous adjacent frame and reduces the number of attack iterations. It is shown that our method can improve attack efficiency greatly while achieving comparable attack performance with traditional methods. Besides, we also extend our method to video recognition models which are higher level vision tasks and achieves great attack efficiency.



## **25. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

cs.CV

20 pages, 6 figures

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2210.14410v2) [paper-pdf](http://arxiv.org/pdf/2210.14410v2)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.



## **26. Towards Adversarial-Resilient Deep Neural Networks for False Data Injection Attack Detection in Power Grids**

cs.CR

This paper has been accepted by the the 32nd International Conference  on Computer Communications and Networks (ICCCN 2023)

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2102.09057v2) [paper-pdf](http://arxiv.org/pdf/2102.09057v2)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Stella Sun, Kevin Tomsovic, Hairong Qi

**Abstract**: False data injection attacks (FDIAs) pose a significant security threat to power system state estimation. To detect such attacks, recent studies have proposed machine learning (ML) techniques, particularly deep neural networks (DNNs). However, most of these methods fail to account for the risk posed by adversarial measurements, which can compromise the reliability of DNNs in various ML applications. In this paper, we present a DNN-based FDIA detection approach that is resilient to adversarial attacks. We first analyze several adversarial defense mechanisms used in computer vision and show their inherent limitations in FDIA detection. We then propose an adversarial-resilient DNN detection framework for FDIA that incorporates random input padding in both the training and inference phases. Our simulations, based on an IEEE standard power system, demonstrate that this framework significantly reduces the effectiveness of adversarial attacks while having a negligible impact on the DNNs' detection performance.



## **27. Invisible Backdoor Attack with Dynamic Triggers against Person Re-identification**

cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2211.10933v2) [paper-pdf](http://arxiv.org/pdf/2211.10933v2)

**Authors**: Wenli Sun, Xinyang Jiang, Shuguang Dou, Dongsheng Li, Duoqian Miao, Cheng Deng, Cairong Zhao

**Abstract**: In recent years, person Re-identification (ReID) has rapidly progressed with wide real-world applications, but also poses significant risks of adversarial attacks. In this paper, we focus on the backdoor attack on deep ReID models. Existing backdoor attack methods follow an all-to-one or all-to-all attack scenario, where all the target classes in the test set have already been seen in the training set. However, ReID is a much more complex fine-grained open-set recognition problem, where the identities in the test set are not contained in the training set. Thus, previous backdoor attack methods for classification are not applicable for ReID. To ameliorate this issue, we propose a novel backdoor attack on deep ReID under a new all-to-unknown scenario, called Dynamic Triggers Invisible Backdoor Attack (DT-IBA). Instead of learning fixed triggers for the target classes from the training set, DT-IBA can dynamically generate new triggers for any unknown identities. Specifically, an identity hashing network is proposed to first extract target identity information from a reference image, which is then injected into the benign images by image steganography. We extensively validate the effectiveness and stealthiness of the proposed attack on benchmark datasets, and evaluate the effectiveness of several defense methods against our attack.



## **28. The Robustness of Computer Vision Models against Common Corruptions: a Survey**

cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.06024v1) [paper-pdf](http://arxiv.org/pdf/2305.06024v1)

**Authors**: Shunxin Wang, Raymond Veldhuis, Nicola Strisciuglio

**Abstract**: The performance of computer vision models is susceptible to unexpected changes in input images when deployed in real scenarios. These changes are referred to as common corruptions. While they can hinder the applicability of computer vision models in real-world scenarios, they are not always considered as a testbed for model generalization and robustness. In this survey, we present a comprehensive and systematic overview of methods that improve corruption robustness of computer vision models. Unlike existing surveys that focus on adversarial attacks and label noise, we cover extensively the study of robustness to common corruptions that can occur when deploying computer vision models to work in practical applications. We describe different types of image corruption and provide the definition of corruption robustness. We then introduce relevant evaluation metrics and benchmark datasets. We categorize methods into four groups. We also cover indirect methods that show improvements in generalization and may improve corruption robustness as a byproduct. We report benchmark results collected from the literature and find that they are not evaluated in a unified manner, making it difficult to compare and analyze. We thus built a unified benchmark framework to obtain directly comparable results on benchmark datasets. Furthermore, we evaluate relevant backbone networks pre-trained on ImageNet using our framework, providing an overview of the base corruption robustness of existing models to help choose appropriate backbones for computer vision tasks. We identify that developing methods to handle a wide range of corruptions and efficiently learn with limited data and computational resources is crucial for future development. Additionally, we highlight the need for further investigation into the relationship among corruption robustness, OOD generalization, and shortcut learning.



## **29. Robust multi-agent coordination via evolutionary generation of auxiliary adversarial attackers**

cs.MA

In: Proceedings of the 37th AAAI Conference on Artificial  Intelligence (AAAI'23), 2023

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.05909v1) [paper-pdf](http://arxiv.org/pdf/2305.05909v1)

**Authors**: Lei Yuan, Zi-Qian Zhang, Ke Xue, Hao Yin, Feng Chen, Cong Guan, Li-He Li, Chao Qian, Yang Yu

**Abstract**: Cooperative multi-agent reinforcement learning (CMARL) has shown to be promising for many real-world applications. Previous works mainly focus on improving coordination ability via solving MARL-specific challenges (e.g., non-stationarity, credit assignment, scalability), but ignore the policy perturbation issue when testing in a different environment. This issue hasn't been considered in problem formulation or efficient algorithm design. To address this issue, we firstly model the problem as a limited policy adversary Dec-POMDP (LPA-Dec-POMDP), where some coordinators from a team might accidentally and unpredictably encounter a limited number of malicious action attacks, but the regular coordinators still strive for the intended goal. Then, we propose Robust Multi-Agent Coordination via Evolutionary Generation of Auxiliary Adversarial Attackers (ROMANCE), which enables the trained policy to encounter diversified and strong auxiliary adversarial attacks during training, thus achieving high robustness under various policy perturbations. Concretely, to avoid the ego-system overfitting to a specific attacker, we maintain a set of attackers, which is optimized to guarantee the attackers high attacking quality and behavior diversity. The goal of quality is to minimize the ego-system coordination effect, and a novel diversity regularizer based on sparse action is applied to diversify the behaviors among attackers. The ego-system is then paired with a population of attackers selected from the maintained attacker set, and alternately trained against the constantly evolving attackers. Extensive experiments on multiple scenarios from SMAC indicate our ROMANCE provides comparable or better robustness and generalization ability than other baselines.



## **30. RNNS: Representation Nearest Neighbor Search Black-Box Attack on Code Models**

cs.CR

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.05896v1) [paper-pdf](http://arxiv.org/pdf/2305.05896v1)

**Authors**: Jie Zhang, Wei Ma, Qiang Hu, Xiaofei Xie, Yves Le Traon, Yang Liu

**Abstract**: Pre-trained code models are mainly evaluated using the in-distribution test data. The robustness of models, i.e., the ability to handle hard unseen data, still lacks evaluation. In this paper, we propose a novel search-based black-box adversarial attack guided by model behaviours for pre-trained programming language models, named Representation Nearest Neighbor Search(RNNS), to evaluate the robustness of Pre-trained PL models. Unlike other black-box adversarial attacks, RNNS uses the model-change signal to guide the search in the space of the variable names collected from real-world projects. Specifically, RNNS contains two main steps, 1) indicate which variable (attack position location) we should attack based on model uncertainty, and 2) search which adversarial tokens we should use for variable renaming according to the model behaviour observations. We evaluate RNNS on 6 code tasks (e.g., clone detection), 3 programming languages (Java, Python, and C), and 3 pre-trained code models: CodeBERT, GraphCodeBERT, and CodeT5. The results demonstrate that RNNS outperforms the state-of-the-art black-box attacking methods (MHM and ALERT) in terms of attack success rate (ASR) and query times (QT). The perturbation of generated adversarial examples from RNNS is smaller than the baselines with respect to the number of replaced variables and the variable length change. Our experiments also show that RNNS is efficient in attacking the defended models and is useful for adversarial training.



## **31. Quantization Aware Attack: Enhancing the Transferability of Adversarial Attacks across Target Models with Different Quantization Bitwidths**

cs.CR

9 pages

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.05875v1) [paper-pdf](http://arxiv.org/pdf/2305.05875v1)

**Authors**: Yulong Yang, Chenhao Lin, Qian Li, Chao Shen, Dawei Zhou, Nannan Wang, Tongliang Liu

**Abstract**: Quantized Neural Networks (QNNs) receive increasing attention in resource-constrained scenarios because of their excellent generalization abilities, but their robustness under realistic black-box adversarial attacks has not been deeply studied, in which the adversary requires to improve the attack capability across target models with unknown quantization bitwidths. One major challenge is that adversarial examples transfer poorly against QNNs with unknown bitwidths because of the quantization shift and gradient misalignment issues. This paper proposes the Quantization Aware Attack to enhance the attack transferability by making the substitute model ``aware of'' the target of attacking models with multiple bitwidths. Specifically, we design a training objective with multiple bitwidths to align the gradient of the substitute model with the target model with different bitwidths and thus mitigate the negative effect of the above two issues. We conduct comprehensive evaluations by performing multiple transfer-based attacks on standard models and defense models with different architectures and quantization bitwidths. Experimental results show that QAA significantly improves the adversarial transferability of the state-of-the-art attacks by 3.4%-20.9% against normally trained models and 3.7%-13.4% against adversarially trained models on average.



## **32. VSMask: Defending Against Voice Synthesis Attack via Real-Time Predictive Perturbation**

cs.SD

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05736v1) [paper-pdf](http://arxiv.org/pdf/2305.05736v1)

**Authors**: Yuanda Wang, Hanqing Guo, Guangjing Wang, Bocheng Chen, Qiben Yan

**Abstract**: Deep learning based voice synthesis technology generates artificial human-like speeches, which has been used in deepfakes or identity theft attacks. Existing defense mechanisms inject subtle adversarial perturbations into the raw speech audios to mislead the voice synthesis models. However, optimizing the adversarial perturbation not only consumes substantial computation time, but it also requires the availability of entire speech. Therefore, they are not suitable for protecting live speech streams, such as voice messages or online meetings. In this paper, we propose VSMask, a real-time protection mechanism against voice synthesis attacks. Different from offline protection schemes, VSMask leverages a predictive neural network to forecast the most effective perturbation for the upcoming streaming speech. VSMask introduces a universal perturbation tailored for arbitrary speech input to shield a real-time speech in its entirety. To minimize the audio distortion within the protected speech, we implement a weight-based perturbation constraint to reduce the perceptibility of the added perturbation. We comprehensively evaluate VSMask protection performance under different scenarios. The experimental results indicate that VSMask can effectively defend against 3 popular voice synthesis models. None of the synthetic voice could deceive the speaker verification models or human ears with VSMask protection. In a physical world experiment, we demonstrate that VSMask successfully safeguards the real-time speech by injecting the perturbation over the air.



## **33. Using Anomaly Detection to Detect Poisoning Attacks in Federated Learning Applications**

cs.LG

We will updated this article soon

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2207.08486v2) [paper-pdf](http://arxiv.org/pdf/2207.08486v2)

**Authors**: Ali Raza, Shujun Li, Kim-Phuc Tran, Ludovic Koehl

**Abstract**: Adversarial attacks such as poisoning attacks have attracted the attention of many machine learning researchers. Traditionally, poisoning attacks attempt to inject adversarial training data in order to manipulate the trained model. In federated learning (FL), data poisoning attacks can be generalized to model poisoning attacks, which cannot be detected by simpler methods due to the lack of access to local training data by the detector. State-of-the-art poisoning attack detection methods for FL have various weaknesses, e.g., the number of attackers has to be known or not high enough, working with i.i.d. data only, and high computational complexity. To overcome above weaknesses, we propose a novel framework for detecting poisoning attacks in FL, which employs a reference model based on a public dataset and an auditor model to detect malicious updates. We implemented a detector based on the proposed framework and using a one-class support vector machine (OC-SVM), which reaches the lowest possible computational complexity O(K) where K is the number of clients. We evaluated our detector's performance against state-of-the-art (SOTA) poisoning attacks for two typical applications of FL: electrocardiograph (ECG) classification and human activity recognition (HAR). Our experimental results validated the performance of our detector over other SOTA detection methods.



## **34. Improving Adversarial Transferability via Intermediate-level Perturbation Decay**

cs.LG

Revision of ICML '23 submission for better clarity

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2304.13410v2) [paper-pdf](http://arxiv.org/pdf/2304.13410v2)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.



## **35. Turning Privacy-preserving Mechanisms against Federated Learning**

cs.LG

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05355v1) [paper-pdf](http://arxiv.org/pdf/2305.05355v1)

**Authors**: Marco Arazzi, Mauro Conti, Antonino Nocera, Stjepan Picek

**Abstract**: Recently, researchers have successfully employed Graph Neural Networks (GNNs) to build enhanced recommender systems due to their capability to learn patterns from the interaction between involved entities. In addition, previous studies have investigated federated learning as the main solution to enable a native privacy-preserving mechanism for the construction of global GNN models without collecting sensitive data into a single computation unit. Still, privacy issues may arise as the analysis of local model updates produced by the federated clients can return information related to sensitive local data. For this reason, experts proposed solutions that combine federated learning with Differential Privacy strategies and community-driven approaches, which involve combining data from neighbor clients to make the individual local updates less dependent on local sensitive data. In this paper, we identify a crucial security flaw in such a configuration, and we design an attack capable of deceiving state-of-the-art defenses for federated learning. The proposed attack includes two operating modes, the first one focusing on convergence inhibition (Adversarial Mode), and the second one aiming at building a deceptive rating injection on the global federated model (Backdoor Mode). The experimental results show the effectiveness of our attack in both its modes, returning on average 60% performance detriment in all the tests on Adversarial Mode and fully effective backdoors in 93% of cases for the tests performed on Backdoor Mode.



## **36. Data Protection and Security Issues With Network Error Logging**

cs.CR

Accepted for SECRYPT'23

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05343v1) [paper-pdf](http://arxiv.org/pdf/2305.05343v1)

**Authors**: Libor Polčák, Kamil Jeřábek

**Abstract**: Network Error Logging helps web server operators detect operational problems in real-time to provide fast and reliable services. This paper analyses Network Error Logging from two angles. Firstly, this paper overviews Network Error Logging from the data protection view. The ePrivacy Directive requires consent for non-essential access to the end devices. Nevertheless, the Network Error Logging design does not allow limiting the tracking to consenting users. Other issues lay in GDPR requirements for transparency and the obligations in the contract between controllers and processors of personal data. Secondly, this paper explains Network Error Logging exploitations to deploy long-time trackers to the victim devices. Even though users should be able to disable Network Error Logging, it is not clear how to do so. Web server operators can mitigate the attack by configuring servers to preventively remove policies that adversaries might have added.



## **37. Attack Named Entity Recognition by Entity Boundary Interference**

cs.CL

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05253v1) [paper-pdf](http://arxiv.org/pdf/2305.05253v1)

**Authors**: Yifei Yang, Hongqiu Wu, Hai Zhao

**Abstract**: Named Entity Recognition (NER) is a cornerstone NLP task while its robustness has been given little attention. This paper rethinks the principles of NER attacks derived from sentence classification, as they can easily violate the label consistency between the original and adversarial NER examples. This is due to the fine-grained nature of NER, as even minor word changes in the sentence can result in the emergence or mutation of any entities, resulting in invalid adversarial examples. To this end, we propose a novel one-word modification NER attack based on a key insight, NER models are always vulnerable to the boundary position of an entity to make their decision. We thus strategically insert a new boundary into the sentence and trigger the Entity Boundary Interference that the victim model makes the wrong prediction either on this boundary word or on other words in the sentence. We call this attack Virtual Boundary Attack (ViBA), which is shown to be remarkably effective when attacking both English and Chinese models with a 70%-90% attack success rate on state-of-the-art language models (e.g. RoBERTa, DeBERTa) and also significantly faster than previous methods.



## **38. Generating Phishing Attacks using ChatGPT**

cs.CR

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05133v1) [paper-pdf](http://arxiv.org/pdf/2305.05133v1)

**Authors**: Sayak Saha Roy, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The ability of ChatGPT to generate human-like responses and understand context has made it a popular tool for conversational agents, content creation, data analysis, and research and innovation. However, its effectiveness and ease of accessibility makes it a prime target for generating malicious content, such as phishing attacks, that can put users at risk. In this work, we identify several malicious prompts that can be provided to ChatGPT to generate functional phishing websites. Through an iterative approach, we find that these phishing websites can be made to imitate popular brands and emulate several evasive tactics that have been known to avoid detection by anti-phishing entities. These attacks can be generated using vanilla ChatGPT without the need of any prior adversarial exploits (jailbreaking).



## **39. Communication-Robust Multi-Agent Learning by Adaptable Auxiliary Multi-Agent Adversary Generation**

cs.LG

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05116v1) [paper-pdf](http://arxiv.org/pdf/2305.05116v1)

**Authors**: Lei Yuan, Feng Chen, Zhongzhang Zhang, Yang Yu

**Abstract**: Communication can promote coordination in cooperative Multi-Agent Reinforcement Learning (MARL). Nowadays, existing works mainly focus on improving the communication efficiency of agents, neglecting that real-world communication is much more challenging as there may exist noise or potential attackers. Thus the robustness of the communication-based policies becomes an emergent and severe issue that needs more exploration. In this paper, we posit that the ego system trained with auxiliary adversaries may handle this limitation and propose an adaptable method of Multi-Agent Auxiliary Adversaries Generation for robust Communication, dubbed MA3C, to obtain a robust communication-based policy. In specific, we introduce a novel message-attacking approach that models the learning of the auxiliary attacker as a cooperative problem under a shared goal to minimize the coordination ability of the ego system, with which every information channel may suffer from distinct message attacks. Furthermore, as naive adversarial training may impede the generalization ability of the ego system, we design an attacker population generation approach based on evolutionary learning. Finally, the ego system is paired with an attacker population and then alternatively trained against the continuously evolving attackers to improve its robustness, meaning that both the ego system and the attackers are adaptable. Extensive experiments on multiple benchmarks indicate that our proposed MA3C provides comparable or better robustness and generalization ability than other baselines.



## **40. Escaping saddle points in zeroth-order optimization: the power of two-point estimators**

math.OC

To appear at ICML 2023

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2209.13555v3) [paper-pdf](http://arxiv.org/pdf/2209.13555v3)

**Authors**: Zhaolin Ren, Yujie Tang, Na Li

**Abstract**: Two-point zeroth order methods are important in many applications of zeroth-order optimization, such as robotics, wind farms, power systems, online optimization, and adversarial robustness to black-box attacks in deep neural networks, where the problem may be high-dimensional and/or time-varying. Most problems in these applications are nonconvex and contain saddle points. While existing works have shown that zeroth-order methods utilizing $\Omega(d)$ function valuations per iteration (with $d$ denoting the problem dimension) can escape saddle points efficiently, it remains an open question if zeroth-order methods based on two-point estimators can escape saddle points. In this paper, we show that by adding an appropriate isotropic perturbation at each iteration, a zeroth-order algorithm based on $2m$ (for any $1 \leq m \leq d$) function evaluations per iteration can not only find $\epsilon$-second order stationary points polynomially fast, but do so using only $\tilde{O}\left(\frac{d}{m\epsilon^{2}\bar{\psi}}\right)$ function evaluations, where $\bar{\psi} \geq \tilde{\Omega}\left(\sqrt{\epsilon}\right)$ is a parameter capturing the extent to which the function of interest exhibits the strict saddle property.



## **41. Less is More: Removing Text-regions Improves CLIP Training Efficiency and Robustness**

cs.CV

10 pages, 8 figures

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05095v1) [paper-pdf](http://arxiv.org/pdf/2305.05095v1)

**Authors**: Liangliang Cao, Bowen Zhang, Chen Chen, Yinfei Yang, Xianzhi Du, Wencong Zhang, Zhiyun Lu, Yantao Zheng

**Abstract**: The CLIP (Contrastive Language-Image Pre-training) model and its variants are becoming the de facto backbone in many applications. However, training a CLIP model from hundreds of millions of image-text pairs can be prohibitively expensive. Furthermore, the conventional CLIP model doesn't differentiate between the visual semantics and meaning of text regions embedded in images. This can lead to non-robustness when the text in the embedded region doesn't match the image's visual appearance. In this paper, we discuss two effective approaches to improve the efficiency and robustness of CLIP training: (1) augmenting the training dataset while maintaining the same number of optimization steps, and (2) filtering out samples that contain text regions in the image. By doing so, we significantly improve the classification and retrieval accuracy on public benchmarks like ImageNet and CoCo. Filtering out images with text regions also protects the model from typographic attacks. To verify this, we build a new dataset named ImageNet with Adversarial Text Regions (ImageNet-Attr). Our filter-based CLIP model demonstrates a top-1 accuracy of 68.78\%, outperforming previous models whose accuracy was all below 50\%.



## **42. Distributed Detection over Blockchain-aided Internet of Things in the Presence of Attacks**

cs.CR

16 pages, 4 figures. This work has been submitted to the IEEE TIFS

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05070v1) [paper-pdf](http://arxiv.org/pdf/2305.05070v1)

**Authors**: Yiming Jiang, Jiangfan Zhang

**Abstract**: Distributed detection over a blockchain-aided Internet of Things (BIoT) network in the presence of attacks is considered, where the integrated blockchain is employed to secure data exchanges over the BIoT as well as data storage at the agents of the BIoT. We consider a general adversary model where attackers jointly exploit the vulnerability of IoT devices and that of the blockchain employed in the BIoT. The optimal attacking strategy which minimizes the Kullback-Leibler divergence is pursued. It can be shown that this optimization problem is nonconvex, and hence it is generally intractable to find the globally optimal solution to such a problem. To overcome this issue, we first propose a relaxation method that can convert the original nonconvex optimization problem into a convex optimization problem, and then the analytic expression for the optimal solution to the relaxed convex optimization problem is derived. The optimal value of the relaxed convex optimization problem provides a detection performance guarantee for the BIoT in the presence of attacks. In addition, we develop a coordinate descent algorithm which is based on a capped water-filling method to solve the relaxed convex optimization problem, and moreover, we show that the convergence of the proposed coordinate descent algorithm can be guaranteed.



## **43. A Survey on AI/ML-Driven Intrusion and Misbehavior Detection in Networked Autonomous Systems: Techniques, Challenges and Opportunities**

cs.NI

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05040v1) [paper-pdf](http://arxiv.org/pdf/2305.05040v1)

**Authors**: Opeyemi Ajibuwa, Bechir Hamdaoui, Attila A. Yavuz

**Abstract**: AI/ML-based intrusion detection systems (IDSs) and misbehavior detection systems (MDSs) have shown great potential in identifying anomalies in the network traffic of networked autonomous systems. Despite the vast research efforts, practical deployments of such systems in the real world have been limited. Although the safety-critical nature of autonomous systems and the vulnerability of learning-based techniques to adversarial attacks are among the potential reasons, the lack of objective evaluation and feasibility assessment metrics is one key reason behind the limited adoption of these systems in practical settings. This survey aims to address the aforementioned limitation by presenting an in-depth analysis of AI/ML-based IDSs/MDSs and establishing baseline metrics relevant to networked autonomous systems. Furthermore, this work thoroughly surveys recent studies in this domain, highlighting the evaluation metrics and gaps in the current literature. It also presents key findings derived from our analysis of the surveyed papers and proposes guidelines for providing AI/ML-based IDS/MDS solution approaches suitable for vehicular network applications. Our work provides researchers and practitioners with the needed tools to evaluate the feasibility of AI/ML-based IDS/MDS techniques in real-world settings, with the aim of facilitating the practical adoption of such techniques in emerging autonomous vehicular systems.



## **44. White-Box Multi-Objective Adversarial Attack on Dialogue Generation**

cs.CL

ACL 2023 main conference long paper

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.03655v2) [paper-pdf](http://arxiv.org/pdf/2305.03655v2)

**Authors**: Yufei Li, Zexin Li, Yingfan Gao, Cong Liu

**Abstract**: Pre-trained transformers are popular in state-of-the-art dialogue generation (DG) systems. Such language models are, however, vulnerable to various adversarial samples as studied in traditional tasks such as text classification, which inspires our curiosity about their robustness in DG systems. One main challenge of attacking DG models is that perturbations on the current sentence can hardly degrade the response accuracy because the unchanged chat histories are also considered for decision-making. Instead of merely pursuing pitfalls of performance metrics such as BLEU, ROUGE, we observe that crafting adversarial samples to force longer generation outputs benefits attack effectiveness -- the generated responses are typically irrelevant, lengthy, and repetitive. To this end, we propose a white-box multi-objective attack method called DGSlow. Specifically, DGSlow balances two objectives -- generation accuracy and length, via a gradient-based multi-objective optimizer and applies an adaptive searching mechanism to iteratively craft adversarial samples with only a few modifications. Comprehensive experiments on four benchmark datasets demonstrate that DGSlow could significantly degrade state-of-the-art DG models with a higher success rate than traditional accuracy-based methods. Besides, our crafted sentences also exhibit strong transferability in attacking other models.



## **45. Understanding Noise-Augmented Training for Randomized Smoothing**

cs.LG

Transactions on Machine Learning Research, 2023

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04746v1) [paper-pdf](http://arxiv.org/pdf/2305.04746v1)

**Authors**: Ambar Pal, Jeremias Sulam

**Abstract**: Randomized smoothing is a technique for providing provable robustness guarantees against adversarial attacks while making minimal assumptions about a classifier. This method relies on taking a majority vote of any base classifier over multiple noise-perturbed inputs to obtain a smoothed classifier, and it remains the tool of choice to certify deep and complex neural network models. Nonetheless, non-trivial performance of such smoothed classifier crucially depends on the base model being trained on noise-augmented data, i.e., on a smoothed input distribution. While widely adopted in practice, it is still unclear how this noisy training of the base classifier precisely affects the risk of the robust smoothed classifier, leading to heuristics and tricks that are poorly understood. In this work we analyze these trade-offs theoretically in a binary classification setting, proving that these common observations are not universal. We show that, without making stronger distributional assumptions, no benefit can be expected from predictors trained with noise-augmentation, and we further characterize distributions where such benefit is obtained. Our analysis has direct implications to the practical deployment of randomized smoothing, and we illustrate some of these via experiments on CIFAR-10 and MNIST, as well as on synthetic datasets.



## **46. Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders**

cs.IR

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04694v1) [paper-pdf](http://arxiv.org/pdf/2305.04694v1)

**Authors**: Sulthana Shams, Douglas Leith

**Abstract**: In practice, users of a Recommender System (RS) fall into a few clusters based on their preferences. In this work, we conduct a systematic study on user-cluster targeted data poisoning attacks on Matrix Factorisation (MF) based RS, where an adversary injects fake users with falsely crafted user-item feedback to promote an item to a specific user cluster. We analyse how user and item feature matrices change after data poisoning attacks and identify the factors that influence the effectiveness of the attack on these feature matrices. We demonstrate that the adversary can easily target specific user clusters with minimal effort and that some items are more susceptible to attacks than others. Our theoretical analysis has been validated by the experimental results obtained from two real-world datasets. Our observations from the study could serve as a motivating point to design a more robust RS.



## **47. StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning**

cs.CV

accepted by CVPR 2023

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2302.09309v2) [paper-pdf](http://arxiv.org/pdf/2302.09309v2)

**Authors**: Yuqian Fu, Yu Xie, Yanwei Fu, Yu-Gang Jiang

**Abstract**: Cross-Domain Few-Shot Learning (CD-FSL) is a recently emerging task that tackles few-shot learning across different domains. It aims at transferring prior knowledge learned on the source dataset to novel target datasets. The CD-FSL task is especially challenged by the huge domain gap between different datasets. Critically, such a domain gap actually comes from the changes of visual styles, and wave-SAN empirically shows that spanning the style distribution of the source data helps alleviate this issue. However, wave-SAN simply swaps styles of two images. Such a vanilla operation makes the generated styles ``real'' and ``easy'', which still fall into the original set of the source styles. Thus, inspired by vanilla adversarial learning, a novel model-agnostic meta Style Adversarial training (StyleAdv) method together with a novel style adversarial attack method is proposed for CD-FSL. Particularly, our style attack method synthesizes both ``virtual'' and ``hard'' adversarial styles for model training. This is achieved by perturbing the original style with the signed style gradients. By continually attacking styles and forcing the model to recognize these challenging adversarial styles, our model is gradually robust to the visual styles, thus boosting the generalization ability for novel target datasets. Besides the typical CNN-based backbone, we also employ our StyleAdv method on large-scale pretrained vision transformer. Extensive experiments conducted on eight various target datasets show the effectiveness of our method. Whether built upon ResNet or ViT, we achieve the new state of the art for CD-FSL. Code is available at https://github.com/lovelyqian/StyleAdv-CDFSL.



## **48. Recent Advances in Reliable Deep Graph Learning: Inherent Noise, Distribution Shift, and Adversarial Attack**

cs.LG

Preprint. 9 pages, 2 figures

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2202.07114v2) [paper-pdf](http://arxiv.org/pdf/2202.07114v2)

**Authors**: Jintang Li, Bingzhe Wu, Chengbin Hou, Guoji Fu, Yatao Bian, Liang Chen, Junzhou Huang, Zibin Zheng

**Abstract**: Deep graph learning (DGL) has achieved remarkable progress in both business and scientific areas ranging from finance and e-commerce to drug and advanced material discovery. Despite the progress, applying DGL to real-world applications faces a series of reliability threats including inherent noise, distribution shift, and adversarial attacks. This survey aims to provide a comprehensive review of recent advances for improving the reliability of DGL algorithms against the above threats. In contrast to prior related surveys which mainly focus on adversarial attacks and defense, our survey covers more reliability-related aspects of DGL, i.e., inherent noise and distribution shift. Additionally, we discuss the relationships among above aspects and highlight some important issues to be explored in future research.



## **49. Toward Adversarial Training on Contextualized Language Representation**

cs.CL

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04557v1) [paper-pdf](http://arxiv.org/pdf/2305.04557v1)

**Authors**: Hongqiu Wu, Yongxiang Liu, Hanwen Shi, Hai Zhao, Min Zhang

**Abstract**: Beyond the success story of adversarial training (AT) in the recent text domain on top of pre-trained language models (PLMs), our empirical study showcases the inconsistent gains from AT on some tasks, e.g. commonsense reasoning, named entity recognition. This paper investigates AT from the perspective of the contextualized language representation outputted by PLM encoders. We find the current AT attacks lean to generate sub-optimal adversarial examples that can fool the decoder part but have a minor effect on the encoder. However, we find it necessary to effectively deviate the latter one to allow AT to gain. Based on the observation, we propose simple yet effective \textit{Contextualized representation-Adversarial Training} (CreAT), in which the attack is explicitly optimized to deviate the contextualized representation of the encoder. It allows a global optimization of adversarial examples that can fool the entire model. We also find CreAT gives rise to a better direction to optimize the adversarial examples, to let them less sensitive to hyperparameters. Compared to AT, CreAT produces consistent performance gains on a wider range of tasks and is proven to be more effective for language pre-training where only the encoder part is kept for downstream tasks. We achieve the new state-of-the-art performances on a series of challenging benchmarks, e.g. AdvGLUE (59.1 $ \rightarrow $ 61.1), HellaSWAG (93.0 $ \rightarrow $ 94.9), ANLI (68.1 $ \rightarrow $ 69.3).



## **50. Privacy-preserving Adversarial Facial Features**

cs.CV

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05391v1) [paper-pdf](http://arxiv.org/pdf/2305.05391v1)

**Authors**: Zhibo Wang, He Wang, Shuaifan Jin, Wenwen Zhang, Jiahui Hu, Yan Wang, Peng Sun, Wei Yuan, Kaixin Liu, Kui Ren

**Abstract**: Face recognition service providers protect face privacy by extracting compact and discriminative facial features (representations) from images, and storing the facial features for real-time recognition. However, such features can still be exploited to recover the appearance of the original face by building a reconstruction network. Although several privacy-preserving methods have been proposed, the enhancement of face privacy protection is at the expense of accuracy degradation. In this paper, we propose an adversarial features-based face privacy protection (AdvFace) approach to generate privacy-preserving adversarial features, which can disrupt the mapping from adversarial features to facial images to defend against reconstruction attacks. To this end, we design a shadow model which simulates the attackers' behavior to capture the mapping function from facial features to images and generate adversarial latent noise to disrupt the mapping. The adversarial features rather than the original features are stored in the server's database to prevent leaked features from exposing facial information. Moreover, the AdvFace requires no changes to the face recognition network and can be implemented as a privacy-enhancing plugin in deployed face recognition systems. Extensive experimental results demonstrate that AdvFace outperforms the state-of-the-art face privacy-preserving methods in defending against reconstruction attacks while maintaining face recognition accuracy.



