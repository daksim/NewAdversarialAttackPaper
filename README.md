# Latest Adversarial Attack Papers
**update at 2022-07-15 06:31:21**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Susceptibility of Continual Learning Against Adversarial Attacks**

cs.LG

18 pages, 13 figures

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.05225v2)

**Authors**: Hikmat Khan, Pir Masoom Shah, Syed Farhan Alam Zaidi, Saif ul Islam

**Abstracts**: The recent advances in continual (incremental or lifelong) learning have concentrated on the prevention of forgetting that can lead to catastrophic consequences, but there are two outstanding challenges that must be addressed. The first is the evaluation of the robustness of the proposed methods. The second is ensuring the security of learned tasks remains largely unexplored. This paper presents a comprehensive study of the susceptibility of the continually learned tasks (including both current and previously learned tasks) that are vulnerable to forgetting. Such vulnerability of tasks against adversarial attacks raises profound issues in data integrity and privacy. We consider the task incremental learning (Task-IL) scenario and explore three regularization-based experiments, three replay-based experiments, and one hybrid technique based on the reply and exemplar approach. We examine the robustness of these methods. In particular, we consider cases where we demonstrate that any class belonging to the current or previously learned tasks is prone to misclassification. Our observations highlight the potential limitations of existing Task-IL approaches. Our empirical study recommends that the research community consider the robustness of the proposed continual learning approaches and invest extensive efforts in mitigating catastrophic forgetting.



## **2. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

cs.CV

Workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2206.06761v3)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.



## **3. Adversarially-Aware Robust Object Detector**

cs.CV

17 pages, 7 figures, ECCV2022 oral paper

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06202v1)

**Authors**: Ziyi Dong, Pengxu Wei, Liang Lin

**Abstracts**: Object detection, as a fundamental computer vision task, has achieved a remarkable progress with the emergence of deep neural networks. Nevertheless, few works explore the adversarial robustness of object detectors to resist adversarial attacks for practical applications in various real-world scenarios. Detectors have been greatly challenged by unnoticeable perturbation, with sharp performance drop on clean images and extremely poor performance on adversarial images. In this work, we empirically explore the model training for adversarial robustness in object detection, which greatly attributes to the conflict between learning clean images and adversarial images. To mitigate this issue, we propose a Robust Detector (RobustDet) based on adversarially-aware convolution to disentangle gradients for model learning on clean and adversarial images. RobustDet also employs the Adversarial Image Discriminator (AID) and Consistent Features with Reconstruction (CFR) to ensure a reliable robustness. Extensive experiments on PASCAL VOC and MS-COCO demonstrate that our model effectively disentangles gradients and significantly enhances the detection robustness with maintaining the detection ability on clean images.



## **4. Interactive Machine Learning: A State of the Art Review**

cs.LG

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06196v1)

**Authors**: Natnael A. Wondimu, Cédric Buche, Ubbo Visser

**Abstracts**: Machine learning has proved useful in many software disciplines, including computer vision, speech and audio processing, natural language processing, robotics and some other fields. However, its applicability has been significantly hampered due its black-box nature and significant resource consumption. Performance is achieved at the expense of enormous computational resource and usually compromising the robustness and trustworthiness of the model. Recent researches have been identifying a lack of interactivity as the prime source of these machine learning problems. Consequently, interactive machine learning (iML) has acquired increased attention of researchers on account of its human-in-the-loop modality and relatively efficient resource utilization. Thereby, a state-of-the-art review of interactive machine learning plays a vital role in easing the effort toward building human-centred models. In this paper, we provide a comprehensive analysis of the state-of-the-art of iML. We analyze salient research works using merit-oriented and application/task oriented mixed taxonomy. We use a bottom-up clustering approach to generate a taxonomy of iML research works. Research works on adversarial black-box attacks and corresponding iML based defense system, exploratory machine learning, resource constrained learning, and iML performance evaluation are analyzed under their corresponding theme in our merit-oriented taxonomy. We have further classified these research works into technical and sectoral categories. Finally, research opportunities that we believe are inspiring for future work in iML are discussed thoroughly.



## **5. On the Robustness of Bayesian Neural Networks to Adversarial Attacks**

cs.LG

arXiv admin note: text overlap with arXiv:2002.04359

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06154v1)

**Authors**: Luca Bortolussi, Ginevra Carbone, Luca Laurenti, Andrea Patane, Guido Sanguinetti, Matthew Wicker

**Abstracts**: Vulnerability to adversarial attacks is one of the principal hurdles to the adoption of deep learning in safety-critical applications. Despite significant efforts, both practical and theoretical, training deep learning models robust to adversarial attacks is still an open problem. In this paper, we analyse the geometry of adversarial attacks in the large-data, overparameterized limit for Bayesian Neural Networks (BNNs). We show that, in the limit, vulnerability to gradient-based attacks arises as a result of degeneracy in the data distribution, i.e., when the data lies on a lower-dimensional submanifold of the ambient space. As a direct consequence, we demonstrate that in this limit BNN posteriors are robust to gradient-based adversarial attacks. Crucially, we prove that the expected gradient of the loss with respect to the BNN posterior distribution is vanishing, even when each neural network sampled from the posterior is vulnerable to gradient-based attacks. Experimental results on the MNIST, Fashion MNIST, and half moons datasets, representing the finite data regime, with BNNs trained with Hamiltonian Monte Carlo and Variational Inference, support this line of arguments, showing that BNNs can display both high accuracy on clean data and robustness to both gradient-based and gradient-free based adversarial attacks.



## **6. Neural Network Robustness as a Verification Property: A Principled Case Study**

cs.LG

11 pages, CAV 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2104.01396v2)

**Authors**: Marco Casadio, Ekaterina Komendantskaya, Matthew L. Daggitt, Wen Kokke, Guy Katz, Guy Amir, Idan Refaeli

**Abstracts**: Neural networks are very successful at detecting patterns in noisy data, and have become the technology of choice in many fields. However, their usefulness is hampered by their susceptibility to adversarial attacks. Recently, many methods for measuring and improving a network's robustness to adversarial perturbations have been proposed, and this growing body of research has given rise to numerous explicit or implicit notions of robustness. Connections between these notions are often subtle, and a systematic comparison between them is missing in the literature. In this paper we begin addressing this gap, by setting up general principles for the empirical analysis and evaluation of a network's robustness as a mathematical property - during the network's training phase, its verification, and after its deployment. We then apply these principles and conduct a case study that showcases the practical benefits of our general approach.



## **7. Perturbation Inactivation Based Adversarial Defense for Face Recognition**

cs.CV

Accepted by IEEE Transactions on Information Forensics & Security  (T-IFS)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06035v1)

**Authors**: Min Ren, Yuhao Zhu, Yunlong Wang, Zhenan Sun

**Abstracts**: Deep learning-based face recognition models are vulnerable to adversarial attacks. To curb these attacks, most defense methods aim to improve the robustness of recognition models against adversarial perturbations. However, the generalization capacities of these methods are quite limited. In practice, they are still vulnerable to unseen adversarial attacks. Deep learning models are fairly robust to general perturbations, such as Gaussian noises. A straightforward approach is to inactivate the adversarial perturbations so that they can be easily handled as general perturbations. In this paper, a plug-and-play adversarial defense method, named perturbation inactivation (PIN), is proposed to inactivate adversarial perturbations for adversarial defense. We discover that the perturbations in different subspaces have different influences on the recognition model. There should be a subspace, called the immune space, in which the perturbations have fewer adverse impacts on the recognition model than in other subspaces. Hence, our method estimates the immune space and inactivates the adversarial perturbations by restricting them to this subspace. The proposed method can be generalized to unseen adversarial perturbations since it does not rely on a specific kind of adversarial attack method. This approach not only outperforms several state-of-the-art adversarial defense methods but also demonstrates a superior generalization capacity through exhaustive experiments. Moreover, the proposed method can be successfully applied to four commercial APIs without additional training, indicating that it can be easily generalized to existing face recognition systems. The source code is available at https://github.com/RenMin1991/Perturbation-Inactivate



## **8. BadHash: Invisible Backdoor Attacks against Deep Hashing with Clean Label**

cs.CV

This paper has been accepted by the 30th ACM International Conference  on Multimedia (MM '22, October 10--14, 2022, Lisboa, Portugal)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.00278v3)

**Authors**: Shengshan Hu, Ziqi Zhou, Yechao Zhang, Leo Yu Zhang, Yifeng Zheng, Yuanyuan HE, Hai Jin

**Abstracts**: Due to its powerful feature learning capability and high efficiency, deep hashing has achieved great success in large-scale image retrieval. Meanwhile, extensive works have demonstrated that deep neural networks (DNNs) are susceptible to adversarial examples, and exploring adversarial attack against deep hashing has attracted many research efforts. Nevertheless, backdoor attack, another famous threat to DNNs, has not been studied for deep hashing yet. Although various backdoor attacks have been proposed in the field of image classification, existing approaches failed to realize a truly imperceptive backdoor attack that enjoys invisible triggers and clean label setting simultaneously, and they also cannot meet the intrinsic demand of image retrieval backdoor. In this paper, we propose BadHash, the first generative-based imperceptible backdoor attack against deep hashing, which can effectively generate invisible and input-specific poisoned images with clean label. Specifically, we first propose a new conditional generative adversarial network (cGAN) pipeline to effectively generate poisoned samples. For any given benign image, it seeks to generate a natural-looking poisoned counterpart with a unique invisible trigger. In order to improve the attack effectiveness, we introduce a label-based contrastive learning network LabCLN to exploit the semantic characteristics of different labels, which are subsequently used for confusing and misleading the target model to learn the embedded trigger. We finally explore the mechanism of backdoor attacks on image retrieval in the hash space. Extensive experiments on multiple benchmark datasets verify that BadHash can generate imperceptible poisoned samples with strong attack ability and transferability over state-of-the-art deep hashing schemes.



## **9. Physical Backdoor Attacks to Lane Detection Systems in Autonomous Driving**

cs.CV

Accepted by ACM MultiMedia 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2203.00858v2)

**Authors**: Xingshuo Han, Guowen Xu, Yuan Zhou, Xuehuan Yang, Jiwei Li, Tianwei Zhang

**Abstracts**: Modern autonomous vehicles adopt state-of-the-art DNN models to interpret the sensor data and perceive the environment. However, DNN models are vulnerable to different types of adversarial attacks, which pose significant risks to the security and safety of the vehicles and passengers. One prominent threat is the backdoor attack, where the adversary can compromise the DNN model by poisoning the training samples. Although lots of effort has been devoted to the investigation of the backdoor attack to conventional computer vision tasks, its practicality and applicability to the autonomous driving scenario is rarely explored, especially in the physical world.   In this paper, we target the lane detection system, which is an indispensable module for many autonomous driving tasks, e.g., navigation, lane switching. We design and realize the first physical backdoor attacks to such system. Our attacks are comprehensively effective against different types of lane detection algorithms. Specifically, we introduce two attack methodologies (poison-annotation and clean-annotation) to generate poisoned samples. With those samples, the trained lane detection model will be infected with the backdoor, and can be activated by common objects (e.g., traffic cones) to make wrong detections, leading the vehicle to drive off the road or onto the opposite lane. Extensive evaluations on public datasets and physical autonomous vehicles demonstrate that our backdoor attacks are effective, stealthy and robust against various defense solutions. Our codes and experimental videos can be found in https://sites.google.com/view/lane-detection-attack/lda.



## **10. PatchZero: Defending against Adversarial Patch Attacks by Detecting and Zeroing the Patch**

cs.CV

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.01795v2)

**Authors**: Ke Xu, Yao Xiao, Zhaoheng Zheng, Kaijie Cai, Ram Nevatia

**Abstracts**: Adversarial patch attacks mislead neural networks by injecting adversarial pixels within a local region. Patch attacks can be highly effective in a variety of tasks and physically realizable via attachment (e.g. a sticker) to the real-world objects. Despite the diversity in attack patterns, adversarial patches tend to be highly textured and different in appearance from natural images. We exploit this property and present PatchZero, a general defense pipeline against white-box adversarial patches without retraining the downstream classifier or detector. Specifically, our defense detects adversaries at the pixel-level and "zeros out" the patch region by repainting with mean pixel values. We further design a two-stage adversarial training scheme to defend against the stronger adaptive attacks. PatchZero achieves SOTA defense performance on the image classification (ImageNet, RESISC45), object detection (PASCAL VOC), and video classification (UCF101) tasks with little degradation in benign performance. In addition, PatchZero transfers to different patch shapes and attack types.



## **11. Game of Trojans: A Submodular Byzantine Approach**

cs.LG

Submitted to GameSec 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.05937v1)

**Authors**: Dinuka Sahabandu, Arezoo Rajabi, Luyao Niu, Bo Li, Bhaskar Ramasubramanian, Radha Poovendran

**Abstracts**: Machine learning models in the wild have been shown to be vulnerable to Trojan attacks during training. Although many detection mechanisms have been proposed, strong adaptive attackers have been shown to be effective against them. In this paper, we aim to answer the questions considering an intelligent and adaptive adversary: (i) What is the minimal amount of instances required to be Trojaned by a strong attacker? and (ii) Is it possible for such an attacker to bypass strong detection mechanisms?   We provide an analytical characterization of adversarial capability and strategic interactions between the adversary and detection mechanism that take place in such models. We characterize adversary capability in terms of the fraction of the input dataset that can be embedded with a Trojan trigger. We show that the loss function has a submodular structure, which leads to the design of computationally efficient algorithms to determine this fraction with provable bounds on optimality. We propose a Submodular Trojan algorithm to determine the minimal fraction of samples to inject a Trojan trigger. To evade detection of the Trojaned model, we model strategic interactions between the adversary and Trojan detection mechanism as a two-player game. We show that the adversary wins the game with probability one, thus bypassing detection. We establish this by proving that output probability distributions of a Trojan model and a clean model are identical when following the Min-Max (MM) Trojan algorithm.   We perform extensive evaluations of our algorithms on MNIST, CIFAR-10, and EuroSAT datasets. The results show that (i) with Submodular Trojan algorithm, the adversary needs to embed a Trojan trigger into a very small fraction of samples to achieve high accuracy on both Trojan and clean samples, and (ii) the MM Trojan algorithm yields a trained Trojan model that evades detection with probability 1.



## **12. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Predictions**

cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2205.01094v3)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.



## **13. Practical Attacks on Machine Learning: A Case Study on Adversarial Windows Malware**

cs.CR

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05548v1)

**Authors**: Luca Demetrio, Battista Biggio, Fabio Roli

**Abstracts**: While machine learning is vulnerable to adversarial examples, it still lacks systematic procedures and tools for evaluating its security in different application contexts. In this article, we discuss how to develop automated and scalable security evaluations of machine learning using practical attacks, reporting a use case on Windows malware detection.



## **14. Improving the Robustness and Generalization of Deep Neural Network with Confidence Threshold Reduction**

cs.LG

Under review

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2206.00913v2)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Deep neural networks are easily attacked by imperceptible perturbation. Presently, adversarial training (AT) is the most effective method to enhance the robustness of the model against adversarial examples. However, because adversarial training solved a min-max value problem, in comparison with natural training, the robustness and generalization are contradictory, i.e., the robustness improvement of the model will decrease the generalization of the model. To address this issue, in this paper, a new concept, namely confidence threshold (CT), is introduced and the reducing of the confidence threshold, known as confidence threshold reduction (CTR), is proven to improve both the generalization and robustness of the model. Specifically, to reduce the CT for natural training (i.e., for natural training with CTR), we propose a mask-guided divergence loss function (MDL) consisting of a cross-entropy loss term and an orthogonal term. The empirical and theoretical analysis demonstrates that the MDL loss improves the robustness and generalization of the model simultaneously for natural training. However, the model robustness improvement of natural training with CTR is not comparable to that of adversarial training. Therefore, for adversarial training, we propose a standard deviation loss function (STD), which minimizes the difference in the probabilities of the wrong categories, to reduce the CT by being integrated into the loss function of adversarial training. The empirical and theoretical analysis demonstrates that the STD based loss function can further improve the robustness of the adversarially trained model on basis of guaranteeing the changeless or slight improvement of the natural accuracy.



## **15. Adversarial Robustness Assessment of NeuroEvolution Approaches**

cs.NE

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05451v1)

**Authors**: Inês Valentim, Nuno Lourenço, Nuno Antunes

**Abstracts**: NeuroEvolution automates the generation of Artificial Neural Networks through the application of techniques from Evolutionary Computation. The main goal of these approaches is to build models that maximize predictive performance, sometimes with an additional objective of minimizing computational complexity. Although the evolved models achieve competitive results performance-wise, their robustness to adversarial examples, which becomes a concern in security-critical scenarios, has received limited attention. In this paper, we evaluate the adversarial robustness of models found by two prominent NeuroEvolution approaches on the CIFAR-10 image classification task: DENSER and NSGA-Net. Since the models are publicly available, we consider white-box untargeted attacks, where the perturbations are bounded by either the L2 or the Linfinity-norm. Similarly to manually-designed networks, our results show that when the evolved models are attacked with iterative methods, their accuracy usually drops to, or close to, zero under both distance metrics. The DENSER model is an exception to this trend, showing some resistance under the L2 threat model, where its accuracy only drops from 93.70% to 18.10% even with iterative attacks. Additionally, we analyzed the impact of pre-processing applied to the data before the first layer of the network. Our observations suggest that some of these techniques can exacerbate the perturbations added to the original inputs, potentially harming robustness. Thus, this choice should not be neglected when automatically designing networks for applications where adversarial attacks are prone to occur.



## **16. A Security-aware and LUT-based CAD Flow for the Physical Synthesis of eASICs**

cs.CR

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05413v1)

**Authors**: Zain UlAbideen, Tiago Diadami Perez, Mayler Martins, Samuel Pagliarini

**Abstracts**: Numerous threats are associated with the globalized integrated circuit (IC) supply chain, such as piracy, reverse engineering, overproduction, and malicious logic insertion. Many obfuscation approaches have been proposed to mitigate these threats by preventing an adversary from fully understanding the IC (or parts of it). The use of reconfigurable elements inside an IC is a known obfuscation technique, either as a coarse grain reconfigurable block (i.e., eFPGA) or as a fine grain element (i.e., FPGA-like look-up tables). This paper presents a security-aware CAD flow that is LUT-based yet still compatible with the standard cell based physical synthesis flow. More precisely, our CAD flow explores the FPGA-ASIC design space and produces heavily obfuscated designs where only small portions of the logic resemble an ASIC. Therefore, we term this specialized solution an "embedded ASIC" (eASIC). Nevertheless, even for heavily LUT-dominated designs, our proposed decomposition and pin swapping algorithms allow for performance gains that enable performance levels that only ASICs would otherwise achieve. On the security side, we have developed novel template-based attacks and also applied existing attacks, both oracle-free and oracle-based. Our security analysis revealed that the obfuscation rate for an SHA-256 study case should be at least 45% for withstanding traditional attacks and at least 80% for withstanding template-based attacks. When the 80\% obfuscated SHA-256 design is physically implemented, it achieves a remarkable frequency of 368MHz in a 65nm commercial technology, whereas its FPGA implementation (in a superior technology) achieves only 77MHz.



## **17. Frequency Domain Model Augmentation for Adversarial Attack**

cs.CV

Accepted by ECCV 2022

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05382v1)

**Authors**: Yuyang Long, Qilong Zhang, Boheng Zeng, Lianli Gao, Xianglong Liu, Jian Zhang, Jingkuan Song

**Abstracts**: For black-box attacks, the gap between the substitute model and the victim model is usually large, which manifests as a weak attack performance. Motivated by the observation that the transferability of adversarial examples can be improved by attacking diverse models simultaneously, model augmentation methods which simulate different models by using transformed images are proposed. However, existing transformations for spatial domain do not translate to significantly diverse augmented models. To tackle this issue, we propose a novel spectrum simulation attack to craft more transferable adversarial examples against both normally trained and defense models. Specifically, we apply a spectrum transformation to the input and thus perform the model augmentation in the frequency domain. We theoretically prove that the transformation derived from frequency domain leads to a diverse spectrum saliency map, an indicator we proposed to reflect the diversity of substitute models. Notably, our method can be generally combined with existing attacks. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our method, \textit{e.g.}, attacking nine state-of-the-art defense models with an average success rate of \textbf{95.4\%}. Our code is available in \url{https://github.com/yuyang-long/SSA}.



## **18. Bi-fidelity Evolutionary Multiobjective Search for Adversarially Robust Deep Neural Architectures**

cs.LG

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05321v1)

**Authors**: Jia Liu, Ran Cheng, Yaochu Jin

**Abstracts**: Deep neural networks have been found vulnerable to adversarial attacks, thus raising potentially concerns in security-sensitive contexts. To address this problem, recent research has investigated the adversarial robustness of deep neural networks from the architectural point of view. However, searching for architectures of deep neural networks is computationally expensive, particularly when coupled with adversarial training process. To meet the above challenge, this paper proposes a bi-fidelity multiobjective neural architecture search approach. First, we formulate the NAS problem for enhancing adversarial robustness of deep neural networks into a multiobjective optimization problem. Specifically, in addition to a low-fidelity performance predictor as the first objective, we leverage an auxiliary-objective -- the value of which is the output of a surrogate model trained with high-fidelity evaluations. Secondly, we reduce the computational cost by combining three performance estimation methods, i.e., parameter sharing, low-fidelity evaluation, and surrogate-based predictor. The effectiveness of the proposed approach is confirmed by extensive experiments conducted on CIFAR-10, CIFAR-100 and SVHN datasets.



## **19. Multitask Learning from Augmented Auxiliary Data for Improving Speech Emotion Recognition**

cs.SD

Under review IEEE Transactions on Affective Computing

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05298v1)

**Authors**: Siddique Latif, Rajib Rana, Sara Khalifa, Raja Jurdak, Björn W. Schuller

**Abstracts**: Despite the recent progress in speech emotion recognition (SER), state-of-the-art systems lack generalisation across different conditions. A key underlying reason for poor generalisation is the scarcity of emotion datasets, which is a significant roadblock to designing robust machine learning (ML) models. Recent works in SER focus on utilising multitask learning (MTL) methods to improve generalisation by learning shared representations. However, most of these studies propose MTL solutions with the requirement of meta labels for auxiliary tasks, which limits the training of SER systems. This paper proposes an MTL framework (MTL-AUG) that learns generalised representations from augmented data. We utilise augmentation-type classification and unsupervised reconstruction as auxiliary tasks, which allow training SER systems on augmented data without requiring any meta labels for auxiliary tasks. The semi-supervised nature of MTL-AUG allows for the exploitation of the abundant unlabelled data to further boost the performance of SER. We comprehensively evaluate the proposed framework in the following settings: (1) within corpus, (2) cross-corpus and cross-language, (3) noisy speech, (4) and adversarial attacks. Our evaluations using the widely used IEMOCAP, MSP-IMPROV, and EMODB datasets show improved results compared to existing state-of-the-art methods.



## **20. "Why do so?" -- A Practical Perspective on Machine Learning Security**

cs.LG

under submission - 18 pages, 3 tables and 4 figures. Long version of  the paper accepted at: New Frontiers of Adversarial Machine Learning@ICML

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05164v1)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Battista Biggio, Katharina Krombholz

**Abstracts**: Despite the large body of academic work on machine learning security, little is known about the occurrence of attacks on machine learning systems in the wild. In this paper, we report on a quantitative study with 139 industrial practitioners. We analyze attack occurrence and concern and evaluate statistical hypotheses on factors influencing threat perception and exposure. Our results shed light on real-world attacks on deployed machine learning. On the organizational level, while we find no predictors for threat exposure in our sample, the amount of implement defenses depends on exposure to threats or expected likelihood to become a target. We also provide a detailed analysis of practitioners' replies on the relevance of individual machine learning attacks, unveiling complex concerns like unreliable decision making, business information leakage, and bias introduction into models. Finally, we find that on the individual level, prior knowledge about machine learning security influences threat perception. Our work paves the way for more research about adversarial machine learning in practice, but yields also insights for regulation and auditing.



## **21. LQG Reference Tracking with Safety and Reachability Guarantees under Unknown False Data Injection Attacks**

eess.SY

13 pages, 4 figures, extended version of a Transactions on Automatic  Control paper

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2103.00387v2)

**Authors**: Zhouchi Li, Luyao Niu, Andrew Clark

**Abstracts**: We investigate a linear quadratic Gaussian (LQG) tracking problem with safety and reachability constraints in the presence of an adversary who mounts an FDI attack on an unknown set of sensors. For each possible set of compromised sensors, we maintain a state estimator disregarding the sensors in that set, and calculate the optimal LQG control input at each time based on this estimate. We propose a control policy which constrains the control input to lie within a fixed distance of the optimal control input corresponding to each state estimate. The control input is obtained at each time step by solving a quadratically constrained quadratic program (QCQP). We prove that our policy can achieve a desired probability of safety and reachability using the barrier certificate method. Our control policy is evaluated via a numerical case study.



## **22. Towards Effective Multi-Label Recognition Attacks via Knowledge Graph Consistency**

cs.CV

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05137v1)

**Authors**: Hassan Mahmood, Ehsan Elhamifar

**Abstracts**: Many real-world applications of image recognition require multi-label learning, whose goal is to find all labels in an image. Thus, robustness of such systems to adversarial image perturbations is extremely important. However, despite a large body of recent research on adversarial attacks, the scope of the existing works is mainly limited to the multi-class setting, where each image contains a single label. We show that the naive extensions of multi-class attacks to the multi-label setting lead to violating label relationships, modeled by a knowledge graph, and can be detected using a consistency verification scheme. Therefore, we propose a graph-consistent multi-label attack framework, which searches for small image perturbations that lead to misclassifying a desired target set while respecting label hierarchies. By extensive experiments on two datasets and using several multi-label recognition models, we show that our method generates extremely successful attacks that, unlike naive multi-label perturbations, can produce model predictions consistent with the knowledge graph.



## **23. RUSH: Robust Contrastive Learning via Randomized Smoothing**

cs.LG

12 pages, 2 figures

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05127v1)

**Authors**: Yijiang Pang, Boyang Liu, Jiayu Zhou

**Abstracts**: Recently, adversarial training has been incorporated in self-supervised contrastive pre-training to augment label efficiency with exciting adversarial robustness. However, the robustness came at a cost of expensive adversarial training. In this paper, we show a surprising fact that contrastive pre-training has an interesting yet implicit connection with robustness, and such natural robustness in the pre trained representation enables us to design a powerful robust algorithm against adversarial attacks, RUSH, that combines the standard contrastive pre-training and randomized smoothing. It boosts both standard accuracy and robust accuracy, and significantly reduces training costs as compared with adversarial training. We use extensive empirical studies to show that the proposed RUSH outperforms robust classifiers from adversarial training, by a significant margin on common benchmarks (CIFAR-10, CIFAR-100, and STL-10) under first-order attacks. In particular, under $\ell_{\infty}$-norm perturbations of size 8/255 PGD attack on CIFAR-10, our model using ResNet-18 as backbone reached 77.8% robust accuracy and 87.9% standard accuracy. Our work has an improvement of over 15% in robust accuracy and a slight improvement in standard accuracy, compared to the state-of-the-arts.



## **24. Physical Passive Patch Adversarial Attacks on Visual Odometry Systems**

cs.CV

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05729v1)

**Authors**: Yaniv Nemcovsky, Matan Yaakoby, Alex M. Bronstein, Chaim Baskin

**Abstracts**: Deep neural networks are known to be susceptible to adversarial perturbations -- small perturbations that alter the output of the network and exist under strict norm limitations. While such perturbations are usually discussed as tailored to a specific input, a universal perturbation can be constructed to alter the model's output on a set of inputs. Universal perturbations present a more realistic case of adversarial attacks, as awareness of the model's exact input is not required. In addition, the universal attack setting raises the subject of generalization to unseen data, where given a set of inputs, the universal perturbations aim to alter the model's output on out-of-sample data. In this work, we study physical passive patch adversarial attacks on visual odometry-based autonomous navigation systems. A visual odometry system aims to infer the relative camera motion between two corresponding viewpoints, and is frequently used by vision-based autonomous navigation systems to estimate their state. For such navigation systems, a patch adversarial perturbation poses a severe security issue, as it can be used to mislead a system onto some collision course. To the best of our knowledge, we show for the first time that the error margin of a visual odometry model can be significantly increased by deploying patch adversarial attacks in the scene. We provide evaluation on synthetic closed-loop drone navigation data and demonstrate that a comparable vulnerability exists in real data. A reference implementation of the proposed method and the reported experiments is provided at https://github.com/patchadversarialattacks/patchadversarialattacks.



## **25. Risk assessment and optimal allocation of security measures under stealthy false data injection attacks**

eess.SY

Accepted for publication at 6th IEEE Conference on Control Technology  and Applications (CCTA). arXiv admin note: substantial text overlap with  arXiv:2106.07071

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04860v1)

**Authors**: Sribalaji C. Anand, André M. H. Teixeira, Anders Ahlén

**Abstracts**: This paper firstly addresses the problem of risk assessment under false data injection attacks on uncertain control systems. We consider an adversary with complete system knowledge, injecting stealthy false data into an uncertain control system. We then use the Value-at-Risk to characterize the risk associated with the attack impact caused by the adversary. The worst-case attack impact is characterized by the recently proposed output-to-output gain. We observe that the risk assessment problem corresponds to an infinite non-convex robust optimization problem. To this end, we use dissipative system theory and the scenario approach to approximate the risk-assessment problem into a convex problem and also provide probabilistic certificates on approximation. Secondly, we consider the problem of security measure allocation. We consider an operator with a constraint on the security budget. Under this constraint, we propose an algorithm to optimally allocate the security measures using the calculated risk such that the resulting Value-at-risk is minimized. Finally, we illustrate the results through a numerical example. The numerical example also illustrates that the security allocation using the Value-at-risk, and the impact on the nominal system may have different outcomes: thereby depicting the benefit of using risk metrics.



## **26. Statistical Detection of Adversarial examples in Blockchain-based Federated Forest In-vehicle Network Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04843v1)

**Authors**: Ibrahim Aliyu, Selinde van Engelenburg, Muhammed Bashir Muazu, Jinsul Kim, Chang Gyoon Lim

**Abstracts**: The internet-of-Vehicle (IoV) can facilitate seamless connectivity between connected vehicles (CV), autonomous vehicles (AV), and other IoV entities. Intrusion Detection Systems (IDSs) for IoV networks can rely on machine learning (ML) to protect the in-vehicle network from cyber-attacks. Blockchain-based Federated Forests (BFFs) could be used to train ML models based on data from IoV entities while protecting the confidentiality of the data and reducing the risks of tampering with the data. However, ML models created this way are still vulnerable to evasion, poisoning, and exploratory attacks using adversarial examples. This paper investigates the impact of various possible adversarial examples on the BFF-IDS. We proposed integrating a statistical detector to detect and extract unknown adversarial samples. By including the unknown detected samples into the dataset of the detector, we augment the BFF-IDS with an additional model to detect original known attacks and the new adversarial inputs. The statistical adversarial detector confidently detected adversarial examples at the sample size of 50 and 100 input samples. Furthermore, the augmented BFF-IDS (BFF-IDS(AUG)) successfully mitigates the adversarial examples with more than 96% accuracy. With this approach, the model will continue to be augmented in a sandbox whenever an adversarial sample is detected and subsequently adopt the BFF-IDS(AUG) as the active security model. Consequently, the proposed integration of the statistical adversarial detector and the subsequent augmentation of the BFF-IDS with detected adversarial samples provides a sustainable security framework against adversarial examples and other unknown attacks.



## **27. Physical Attack on Monocular Depth Estimation with Optimal Adversarial Patches**

cs.CV

ECCV2022

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04718v1)

**Authors**: Zhiyuan Cheng, James Liang, Hongjun Choi, Guanhong Tao, Zhiwen Cao, Dongfang Liu, Xiangyu Zhang

**Abstracts**: Deep learning has substantially boosted the performance of Monocular Depth Estimation (MDE), a critical component in fully vision-based autonomous driving (AD) systems (e.g., Tesla and Toyota). In this work, we develop an attack against learning-based MDE. In particular, we use an optimization-based method to systematically generate stealthy physical-object-oriented adversarial patches to attack depth estimation. We balance the stealth and effectiveness of our attack with object-oriented adversarial design, sensitive region localization, and natural style camouflage. Using real-world driving scenarios, we evaluate our attack on concurrent MDE models and a representative downstream task for AD (i.e., 3D object detection). Experimental results show that our method can generate stealthy, effective, and robust adversarial patches for different target objects and models and achieves more than 6 meters mean depth estimation error and 93% attack success rate (ASR) in object detection with a patch of 1/9 of the vehicle's rear area. Field tests on three different driving routes with a real vehicle indicate that we cause over 6 meters mean depth estimation error and reduce the object detection rate from 90.70% to 5.16% in continuous video frames.



## **28. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

cs.CV

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2202.07054v2)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset are available online (https://github.com/YonghaoXu/UAE-RS).



## **29. Visual explanation of black-box model: Similarity Difference and Uniqueness (SIDU) method**

cs.CV

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2101.10710v2)

**Authors**: Satya M. Muddamsetty, Mohammad N. S. Jahromi, Andreea E. Ciontos, Laura M. Fenoy, Thomas B. Moeslund

**Abstracts**: Explainable Artificial Intelligence (XAI) has in recent years become a well-suited framework to generate human understandable explanations of "black-box" models. In this paper, a novel XAI visual explanation algorithm known as the Similarity Difference and Uniqueness (SIDU) method that can effectively localize entire object regions responsible for prediction is presented in full detail. The SIDU algorithm robustness and effectiveness is analyzed through various computational and human subject experiments. In particular, the SIDU algorithm is assessed using three different types of evaluations (Application, Human and Functionally-Grounded) to demonstrate its superior performance. The robustness of SIDU is further studied in the presence of adversarial attack on "black-box" models to better understand its performance. Our code is available at: https://github.com/satyamahesh84/SIDU_XAI_CODE.



## **30. Fooling Partial Dependence via Data Poisoning**

cs.LG

Accepted at ECML PKDD 2022

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2105.12837v3)

**Authors**: Hubert Baniecki, Wojciech Kretowicz, Przemyslaw Biecek

**Abstracts**: Many methods have been developed to understand complex predictive models and high expectations are placed on post-hoc model explainability. It turns out that such explanations are not robust nor trustworthy, and they can be fooled. This paper presents techniques for attacking Partial Dependence (plots, profiles, PDP), which are among the most popular methods of explaining any predictive model trained on tabular data. We showcase that PD can be manipulated in an adversarial manner, which is alarming, especially in financial or medical applications where auditability became a must-have trait supporting black-box machine learning. The fooling is performed via poisoning the data to bend and shift explanations in the desired direction using genetic and gradient algorithms. We believe this to be the first work using a genetic algorithm for manipulating explanations, which is transferable as it generalizes both ways: in a model-agnostic and an explanation-agnostic manner.



## **31. Adversarial Framework with Certified Robustness for Time-Series Domain via Statistical Features**

cs.LG

Published at Journal of Artificial Intelligence Research

**SubmitDate**: 2022-07-09    [paper-pdf](http://arxiv.org/pdf/2207.04307v1)

**Authors**: Taha Belkhouja, Janardhan Rao Doppa

**Abstracts**: Time-series data arises in many real-world applications (e.g., mobile health) and deep neural networks (DNNs) have shown great success in solving them. Despite their success, little is known about their robustness to adversarial attacks. In this paper, we propose a novel adversarial framework referred to as Time-Series Attacks via STATistical Features (TSA-STAT)}. To address the unique challenges of time-series domain, TSA-STAT employs constraints on statistical features of the time-series data to construct adversarial examples. Optimized polynomial transformations are used to create attacks that are more effective (in terms of successfully fooling DNNs) than those based on additive perturbations. We also provide certified bounds on the norm of the statistical features for constructing adversarial examples. Our experiments on diverse real-world benchmark datasets show the effectiveness of TSA-STAT in fooling DNNs for time-series domain and in improving their robustness. The source code of TSA-STAT algorithms is available at https://github.com/tahabelkhouja/Time-Series-Attacks-via-STATistical-Features



## **32. Not all broken defenses are equal: The dead angles of adversarial accuracy**

cs.LG

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2207.04129v1)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Robustness to adversarial attack is typically evaluated with adversarial accuracy. This metric is however too coarse to properly capture all robustness properties of machine learning models. Many defenses, when evaluated against a strong attack, do not provide accuracy improvements while still contributing partially to adversarial robustness. Popular certification methods suffer from the same issue, as they provide a lower bound to accuracy. To capture finer robustness properties we propose a new metric for L2 robustness, adversarial angular sparsity, which partially answers the question "how many adversarial examples are there around an input". We demonstrate its usefulness by evaluating both "strong" and "weak" defenses. We show that some state-of-the-art defenses, delivering very similar accuracy, can have very different sparsity on the inputs that they are not robust on. We also show that some weak defenses actually decrease robustness, while others strengthen it in a measure that accuracy cannot capture. These differences are predictive of how useful such defenses can become when combined with adversarial training.



## **33. Neighbors From Hell: Voltage Attacks Against Deep Learning Accelerators on Multi-Tenant FPGAs**

cs.CR

Published in the 2020 proceedings of the International Conference of  Field-Programmable Technology (ICFPT)

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2012.07242v2)

**Authors**: Andrew Boutros, Mathew Hall, Nicolas Papernot, Vaughn Betz

**Abstracts**: Field-programmable gate arrays (FPGAs) are becoming widely used accelerators for a myriad of datacenter applications due to their flexibility and energy efficiency. Among these applications, FPGAs have shown promising results in accelerating low-latency real-time deep learning (DL) inference, which is becoming an indispensable component of many end-user applications. With the emerging research direction towards virtualized cloud FPGAs that can be shared by multiple users, the security aspect of FPGA-based DL accelerators requires careful consideration. In this work, we evaluate the security of DL accelerators against voltage-based integrity attacks in a multitenant FPGA scenario. We first demonstrate the feasibility of such attacks on a state-of-the-art Stratix 10 card using different attacker circuits that are logically and physically isolated in a separate attacker role, and cannot be flagged as malicious circuits by conventional bitstream checkers. We show that aggressive clock gating, an effective power-saving technique, can also be a potential security threat in modern FPGAs. Then, we carry out the attack on a DL accelerator running ImageNet classification in the victim role to evaluate the inherent resilience of DL models against timing faults induced by the adversary. We find that even when using the strongest attacker circuit, the prediction accuracy of the DL accelerator is not compromised when running at its safe operating frequency. Furthermore, we can achieve 1.18-1.31x higher inference performance by over-clocking the DL accelerator without affecting its prediction accuracy.



## **34. Defense Against Multi-target Trojan Attacks**

cs.CV

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2207.03895v1)

**Authors**: Haripriya Harikumar, Santu Rana, Kien Do, Sunil Gupta, Wei Zong, Willy Susilo, Svetha Venkastesh

**Abstracts**: Adversarial attacks on deep learning-based models pose a significant threat to the current AI infrastructure. Among them, Trojan attacks are the hardest to defend against. In this paper, we first introduce a variation of the Badnet kind of attacks that introduces Trojan backdoors to multiple target classes and allows triggers to be placed anywhere in the image. The former makes it more potent and the latter makes it extremely easy to carry out the attack in the physical space. The state-of-the-art Trojan detection methods fail with this threat model. To defend against this attack, we first introduce a trigger reverse-engineering mechanism that uses multiple images to recover a variety of potential triggers. We then propose a detection mechanism by measuring the transferability of such recovered triggers. A Trojan trigger will have very high transferability i.e. they make other images also go to the same class. We study many practical advantages of our attack method and then demonstrate the detection performance using a variety of image datasets. The experimental results show the superior detection performance of our method over the state-of-the-arts.



## **35. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

cs.CR

Accepted to ECCV 2022

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2202.12154v4)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a single input-agnostic trigger and targeting only one class to using multiple, input-specific triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make inadequate assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. To deal with this problem, we propose two novel "filtering" defenses called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF) which leverage lossy data compression and adversarial learning respectively to effectively purify potential Trojan triggers in the input at run time without making assumptions about the number of triggers/target classes or the input dependence property of triggers. In addition, we introduce a new defense mechanism called "Filtering-then-Contrasting" (FtC) which helps avoid the drop in classification accuracy on clean data caused by "filtering", and combine it with VIF/AIF to derive new defenses of this kind. Extensive experimental results and ablation studies show that our proposed defenses significantly outperform well-known baseline defenses in mitigating five advanced Trojan attacks including two recent state-of-the-art while being quite robust to small amounts of training data and large-norm triggers.



## **36. On the Relationship Between Adversarial Robustness and Decision Region in Deep Neural Network**

cs.LG

14 pages

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2207.03400v1)

**Authors**: Seongjin Park, Haedong Jeong, Giyoung Jeon, Jaesik Choi

**Abstracts**: In general, Deep Neural Networks (DNNs) are evaluated by the generalization performance measured on unseen data excluded from the training phase. Along with the development of DNNs, the generalization performance converges to the state-of-the-art and it becomes difficult to evaluate DNNs solely based on this metric. The robustness against adversarial attack has been used as an additional metric to evaluate DNNs by measuring their vulnerability. However, few studies have been performed to analyze the adversarial robustness in terms of the geometry in DNNs. In this work, we perform an empirical study to analyze the internal properties of DNNs that affect model robustness under adversarial attacks. In particular, we propose the novel concept of the Populated Region Set (PRS), where training samples are populated more frequently, to represent the internal properties of DNNs in a practical setting. From systematic experiments with the proposed concept, we provide empirical evidence to validate that a low PRS ratio has a strong relationship with the adversarial robustness of DNNs. We also devise PRS regularizer leveraging the characteristics of PRS to improve the adversarial robustness without adversarial training.



## **37. Federated Robustness Propagation: Sharing Robustness in Heterogeneous Federated Learning**

cs.LG

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2106.10196v2)

**Authors**: Junyuan Hong, Haotao Wang, Zhangyang Wang, Jiayu Zhou

**Abstracts**: Federated learning (FL) emerges as a popular distributed learning schema that learns a model from a set of participating users without sharing raw data. One major challenge of FL comes with heterogeneous users, who may have distributionally different (or non-iid) data and varying computation resources. As federated users would use the model for prediction, they often demand the trained model to be robust against malicious attackers at test time. Whereas adversarial training (AT) provides a sound solution for centralized learning, extending its usage for federated users has imposed significant challenges, as many users may have very limited training data and tight computational budgets, to afford the data-hungry and costly AT. In this paper, we study a novel FL strategy: propagating adversarial robustness from rich-resource users that can afford AT, to those with poor resources that cannot afford it, during federated learning. We show that existing FL techniques cannot be effectively integrated with the strategy to propagate robustness among non-iid users and propose an efficient propagation approach by the proper use of batch-normalization. We demonstrate the rationality and effectiveness of our method through extensive experiments. Especially, the proposed method is shown to grant federated models remarkable robustness even when only a small portion of users afford AT during learning. Source code will be released.



## **38. SYNFI: Pre-Silicon Fault Analysis of an Open-Source Secure Element**

cs.CR

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2205.04775v2)

**Authors**: Pascal Nasahl, Miguel Osorio, Pirmin Vogel, Michael Schaffner, Timothy Trippel, Dominic Rizzo, Stefan Mangard

**Abstracts**: Fault attacks are active, physical attacks that an adversary can leverage to alter the control-flow of embedded devices to gain access to sensitive information or bypass protection mechanisms. Due to the severity of these attacks, manufacturers deploy hardware-based fault defenses into security-critical systems, such as secure elements. The development of these countermeasures is a challenging task due to the complex interplay of circuit components and because contemporary design automation tools tend to optimize inserted structures away, thereby defeating their purpose. Hence, it is critical that such countermeasures are rigorously verified post-synthesis. As classical functional verification techniques fall short of assessing the effectiveness of countermeasures, developers have to resort to methods capable of injecting faults in a simulation testbench or into a physical chip. However, developing test sequences to inject faults in simulation is an error-prone task and performing fault attacks on a chip requires specialized equipment and is incredibly time-consuming. To that end, this paper introduces SYNFI, a formal pre-silicon fault verification framework that operates on synthesized netlists. SYNFI can be used to analyze the general effect of faults on the input-output relationship in a circuit and its fault countermeasures, and thus enables hardware designers to assess and verify the effectiveness of embedded countermeasures in a systematic and semi-automatic way. To demonstrate that SYNFI is capable of handling unmodified, industry-grade netlists synthesized with commercial and open tools, we analyze OpenTitan, the first open-source secure element. In our analysis, we identified critical security weaknesses in the unprotected AES block, developed targeted countermeasures, reassessed their security, and contributed these countermeasures back to the OpenTitan repository.



## **39. The StarCraft Multi-Agent Challenges+ : Learning of Multi-Stage Tasks and Environmental Factors without Precise Reward Functions**

cs.LG

ICML Workshop: AI for Agent Based Modeling 2022 Spotlight

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2207.02007v2)

**Authors**: Mingyu Kim, Jihwan Oh, Yongsik Lee, Joonkee Kim, Seonghwan Kim, Song Chong, Se-Young Yun

**Abstracts**: In this paper, we propose a novel benchmark called the StarCraft Multi-Agent Challenges+, where agents learn to perform multi-stage tasks and to use environmental factors without precise reward functions. The previous challenges (SMAC) recognized as a standard benchmark of Multi-Agent Reinforcement Learning are mainly concerned with ensuring that all agents cooperatively eliminate approaching adversaries only through fine manipulation with obvious reward functions. This challenge, on the other hand, is interested in the exploration capability of MARL algorithms to efficiently learn implicit multi-stage tasks and environmental factors as well as micro-control. This study covers both offensive and defensive scenarios. In the offensive scenarios, agents must learn to first find opponents and then eliminate them. The defensive scenarios require agents to use topographic features. For example, agents need to position themselves behind protective structures to make it harder for enemies to attack. We investigate MARL algorithms under SMAC+ and observe that recent approaches work well in similar settings to the previous challenges, but misbehave in offensive scenarios. Additionally, we observe that an enhanced exploration approach has a positive effect on performance but is not able to completely solve all scenarios. This study proposes new directions for future research.



## **40. The Weaknesses of Adversarial Camouflage in Overhead Imagery**

cs.CV

7 pages, 15 figures

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02963v1)

**Authors**: Adam Van Etten

**Abstracts**: Machine learning is increasingly critical for analysis of the ever-growing corpora of overhead imagery. Advanced computer vision object detection techniques have demonstrated great success in identifying objects of interest such as ships, automobiles, and aircraft from satellite and drone imagery. Yet relying on computer vision opens up significant vulnerabilities, namely, the susceptibility of object detection algorithms to adversarial attacks. In this paper we explore the efficacy and drawbacks of adversarial camouflage in an overhead imagery context. While a number of recent papers have demonstrated the ability to reliably fool deep learning classifiers and object detectors with adversarial patches, most of this work has been performed on relatively uniform datasets and only a single class of objects. In this work we utilize the VisDrone dataset, which has a large range of perspectives and object sizes. We explore four different object classes: bus, car, truck, van. We build a library of 24 adversarial patches to disguise these objects, and introduce a patch translucency variable to our patches. The translucency (or alpha value) of the patches is highly correlated to their efficacy. Further, we show that while adversarial patches may fool object detectors, the presence of such patches is often easily uncovered, with patches on average 24% more detectable than the objects the patches were meant to hide. This raises the question of whether such patches truly constitute camouflage. Source code is available at https://github.com/IQTLabs/camolo.



## **41. DeepAdversaries: Examining the Robustness of Deep Learning Models for Galaxy Morphology Classification**

cs.LG

20 pages, 6 figures, 5 tables; accepted in MLST

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2112.14299v3)

**Authors**: Aleksandra Ćiprijanović, Diana Kafkes, Gregory Snyder, F. Javier Sánchez, Gabriel Nathan Perdue, Kevin Pedro, Brian Nord, Sandeep Madireddy, Stefan M. Wild

**Abstracts**: With increased adoption of supervised deep learning methods for processing and analysis of cosmological survey data, the assessment of data perturbation effects (that can naturally occur in the data processing and analysis pipelines) and the development of methods that increase model robustness are increasingly important. In the context of morphological classification of galaxies, we study the effects of perturbations in imaging data. In particular, we examine the consequences of using neural networks when training on baseline data and testing on perturbed data. We consider perturbations associated with two primary sources: 1) increased observational noise as represented by higher levels of Poisson noise and 2) data processing noise incurred by steps such as image compression or telescope errors as represented by one-pixel adversarial attacks. We also test the efficacy of domain adaptation techniques in mitigating the perturbation-driven errors. We use classification accuracy, latent space visualizations, and latent space distance to assess model robustness. Without domain adaptation, we find that processing pixel-level errors easily flip the classification into an incorrect class and that higher observational noise makes the model trained on low-noise data unable to classify galaxy morphologies. On the other hand, we show that training with domain adaptation improves model robustness and mitigates the effects of these perturbations, improving the classification accuracy by 23% on data with higher observational noise. Domain adaptation also increases by a factor of ~2.3 the latent space distance between the baseline and the incorrectly classified one-pixel perturbed image, making the model more robust to inadvertent perturbations.



## **42. Enhancing Adversarial Attacks on Single-Layer NVM Crossbar-Based Neural Networks with Power Consumption Information**

cs.LG

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02764v1)

**Authors**: Cory Merkel

**Abstracts**: Adversarial attacks on state-of-the-art machine learning models pose a significant threat to the safety and security of mission-critical autonomous systems. This paper considers the additional vulnerability of machine learning models when attackers can measure the power consumption of their underlying hardware platform. In particular, we explore the utility of power consumption information for adversarial attacks on non-volatile memory crossbar-based single-layer neural networks. Our results from experiments with MNIST and CIFAR-10 datasets show that power consumption can reveal important information about the neural network's weight matrix, such as the 1-norm of its columns. That information can be used to infer the sensitivity of the network's loss with respect to different inputs. We also find that surrogate-based black box attacks that utilize crossbar power information can lead to improved attack efficiency.



## **43. LADDER: Latent Boundary-guided Adversarial Training**

cs.LG

To appear in Machine Learning

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2206.03717v2)

**Authors**: Xiaowei Zhou, Ivor W. Tsang, Jie Yin

**Abstracts**: Deep Neural Networks (DNNs) have recently achieved great success in many classification tasks. Unfortunately, they are vulnerable to adversarial attacks that generate adversarial examples with a small perturbation to fool DNN models, especially in model sharing scenarios. Adversarial training is proved to be the most effective strategy that injects adversarial examples into model training to improve the robustness of DNN models against adversarial attacks. However, adversarial training based on the existing adversarial examples fails to generalize well to standard, unperturbed test data. To achieve a better trade-off between standard accuracy and adversarial robustness, we propose a novel adversarial training framework called LAtent bounDary-guided aDvErsarial tRaining (LADDER) that adversarially trains DNN models on latent boundary-guided adversarial examples. As opposed to most of the existing methods that generate adversarial examples in the input space, LADDER generates a myriad of high-quality adversarial examples through adding perturbations to latent features. The perturbations are made along the normal of the decision boundary constructed by an SVM with an attention mechanism. We analyze the merits of our generated boundary-guided adversarial examples from a boundary field perspective and visualization view. Extensive experiments and detailed analysis on MNIST, SVHN, CelebA, and CIFAR-10 validate the effectiveness of LADDER in achieving a better trade-off between standard accuracy and adversarial robustness as compared with vanilla DNNs and competitive baselines.



## **44. Adversarial Robustness of Visual Dialog**

cs.CV

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02639v1)

**Authors**: Lu Yu, Verena Rieser

**Abstracts**: Adversarial robustness evaluates the worst-case performance scenario of a machine learning model to ensure its safety and reliability. This study is the first to investigate the robustness of visually grounded dialog models towards textual attacks. These attacks represent a worst-case scenario where the input question contains a synonym which causes the previously correct model to return a wrong answer. Using this scenario, we first aim to understand how multimodal input components contribute to model robustness. Our results show that models which encode dialog history are more robust, and when launching an attack on history, model prediction becomes more uncertain. This is in contrast to prior work which finds that dialog history is negligible for model performance on this task. We also evaluate how to generate adversarial test examples which successfully fool the model but remain undetected by the user/software designer. We find that the textual, as well as the visual context are important to generate plausible worst-case scenarios.



## **45. Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Models**

cs.CV

16

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2111.10759v2)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical universal adversarial perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask's effectiveness in real-world experiments (CCTV use case) by printing the adversarial pattern on a fabric face mask. In these experiments, the FR system was only able to identify 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks). A demo of our experiments can be found at: https://youtu.be/_TXkDO5z11w.



## **46. FIDO2 With Two Displays-Or How to Protect Security-Critical Web Transactions Against Malware Attacks**

cs.CR

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2206.13358v3)

**Authors**: Timon Hackenjos, Benedikt Wagner, Julian Herr, Jochen Rill, Marek Wehmer, Niklas Goerke, Ingmar Baumgart

**Abstracts**: With the rise of attacks on online accounts in the past years, more and more services offer two-factor authentication for their users. Having factors out of two of the three categories something you know, something you have and something you are should ensure that an attacker cannot compromise two of them at once. Thus, an adversary should not be able to maliciously interact with one's account. However, this is only true if one considers a weak adversary. In particular, since most current solutions only authenticate a session and not individual transactions, they are noneffective if one's device is infected with malware. For online banking, the banking industry has long since identified the need for authenticating transactions. However, specifications of such authentication schemes are not public and implementation details vary wildly from bank to bank with most still being unable to protect against malware. In this work, we present a generic approach to tackle the problem of malicious account takeovers, even in the presence of malware. To this end, we define a new paradigm to improve two-factor authentication that involves the concepts of one-out-of-two security and transaction authentication. Web authentication schemes following this paradigm can protect security-critical transactions against manipulation, even if one of the factors is completely compromised. Analyzing existing authentication schemes, we find that they do not realize one-out-of-two security. We give a blueprint of how to design secure web authentication schemes in general. Based on this blueprint we propose FIDO2 With Two Displays (FIDO2D), a new web authentication scheme based on the FIDO2 standard and prove its security using Tamarin. We hope that our work inspires a new wave of more secure web authentication schemes, which protect security-critical transactions even against attacks with malware.



## **47. Learning to Accelerate Approximate Methods for Solving Integer Programming via Early Fixing**

cs.DM

16 pages, 11 figures, 6 tables

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.02087v1)

**Authors**: Longkang Li, Baoyuan Wu

**Abstracts**: Integer programming (IP) is an important and challenging problem. Approximate methods have shown promising performance on both effectiveness and efficiency for solving the IP problem. However, we observed that a large fraction of variables solved by some iterative approximate methods fluctuate around their final converged discrete states in very long iterations. Inspired by this observation, we aim to accelerate these approximate methods by early fixing these fluctuated variables to their converged states while not significantly harming the solution accuracy. To this end, we propose an early fixing framework along with the approximate method. We formulate the whole early fixing process as a Markov decision process, and train it using imitation learning. A policy network will evaluate the posterior probability of each free variable concerning its discrete candidate states in each block of iterations. Specifically, we adopt the powerful multi-headed attention mechanism in the policy network. Extensive experiments on our proposed early fixing framework are conducted to three different IP applications: constrained linear programming, MRF energy minimization and sparse adversarial attack. The former one is linear IP problem, while the latter two are quadratic IP problems. We extend the problem scale from regular size to significantly large size. The extensive experiments reveal the competitiveness of our early fixing framework: the runtime speeds up significantly, while the solution quality does not degrade much, even in some cases it is available to obtain better solutions. Our proposed early fixing framework can be regarded as an acceleration extension of ADMM methods for solving integer programming. The source codes are available at \url{https://github.com/SCLBD/Accelerated-Lpbox-ADMM}.



## **48. Benign Adversarial Attack: Tricking Models for Goodness**

cs.AI

ACM MM2022 Brave New Idea

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2107.11986v2)

**Authors**: Jitao Sang, Xian Zhao, Jiaming Zhang, Zhiyu Lin

**Abstracts**: In spite of the successful application in many fields, machine learning models today suffer from notorious problems like vulnerability to adversarial examples. Beyond falling into the cat-and-mouse game between adversarial attack and defense, this paper provides alternative perspective to consider adversarial example and explore whether we can exploit it in benign applications. We first attribute adversarial example to the human-model disparity on employing non-semantic features. While largely ignored in classical machine learning mechanisms, non-semantic feature enjoys three interesting characteristics as (1) exclusive to model, (2) critical to affect inference, and (3) utilizable as features. Inspired by this, we present brave new idea of benign adversarial attack to exploit adversarial examples for goodness in three directions: (1) adversarial Turing test, (2) rejecting malicious model application, and (3) adversarial data augmentation. Each direction is positioned with motivation elaboration, justification analysis and prototype applications to showcase its potential.



## **49. Formalizing and Estimating Distribution Inference Risks**

cs.LG

Update: Accepted at PETS 2022

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2109.06024v6)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks. Code is available at https://github.com/iamgroot42/FormEstDistRisks



## **50. Query-Efficient Adversarial Attack Based on Latin Hypercube Sampling**

cs.CV

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.02391v1)

**Authors**: Dan Wang, Jiayu Lin, Yuan-Gen Wang

**Abstracts**: In order to be applicable in real-world scenario, Boundary Attacks (BAs) were proposed and ensured one hundred percent attack success rate with only decision information. However, existing BA methods craft adversarial examples by leveraging a simple random sampling (SRS) to estimate the gradient, consuming a large number of model queries. To overcome the drawback of SRS, this paper proposes a Latin Hypercube Sampling based Boundary Attack (LHS-BA) to save query budget. Compared with SRS, LHS has better uniformity under the same limited number of random samples. Therefore, the average on these random samples is closer to the true gradient than that estimated by SRS. Various experiments are conducted on benchmark datasets including MNIST, CIFAR, and ImageNet-1K. Experimental results demonstrate the superiority of the proposed LHS-BA over the state-of-the-art BA methods in terms of query efficiency. The source codes are publicly available at https://github.com/GZHU-DVL/LHS-BA.



