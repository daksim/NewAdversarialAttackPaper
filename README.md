# Latest Adversarial Attack Papers
**update at 2023-03-29 16:29:53**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. A Survey on Malware Detection with Graph Representation Learning**

cs.CR

Preprint, submitted to ACM Computing Surveys on March 2023. For any  suggestions or improvements, please contact me directly by e-mail

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.16004v1) [paper-pdf](http://arxiv.org/pdf/2303.16004v1)

**Authors**: Tristan Bilot, Nour El Madhoun, Khaldoun Al Agha, Anis Zouaoui

**Abstract**: Malware detection has become a major concern due to the increasing number and complexity of malware. Traditional detection methods based on signatures and heuristics are used for malware detection, but unfortunately, they suffer from poor generalization to unknown attacks and can be easily circumvented using obfuscation techniques. In recent years, Machine Learning (ML) and notably Deep Learning (DL) achieved impressive results in malware detection by learning useful representations from data and have become a solution preferred over traditional methods. More recently, the application of such techniques on graph-structured data has achieved state-of-the-art performance in various domains and demonstrates promising results in learning more robust representations from malware. Yet, no literature review focusing on graph-based deep learning for malware detection exists. In this survey, we provide an in-depth literature review to summarize and unify existing works under the common approaches and architectures. We notably demonstrate that Graph Neural Networks (GNNs) reach competitive results in learning robust embeddings from malware represented as expressive graph structures, leading to an efficient detection by downstream classifiers. This paper also reviews adversarial attacks that are utilized to fool graph-based detection methods. Challenges and future research directions are discussed at the end of the paper.



## **2. TransAudio: Towards the Transferable Adversarial Audio Attack via Learning Contextualized Perturbations**

cs.SD

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15940v1) [paper-pdf](http://arxiv.org/pdf/2303.15940v1)

**Authors**: Qi Gege, Yuefeng Chen, Xiaofeng Mao, Yao Zhu, Binyuan Hui, Xiaodan Li, Rong Zhang, Hui Xue

**Abstract**: In a transfer-based attack against Automatic Speech Recognition (ASR) systems, attacks are unable to access the architecture and parameters of the target model. Existing attack methods are mostly investigated in voice assistant scenarios with restricted voice commands, prohibiting their applicability to more general ASR related applications. To tackle this challenge, we propose a novel contextualized attack with deletion, insertion, and substitution adversarial behaviors, namely TransAudio, which achieves arbitrary word-level attacks based on the proposed two-stage framework. To strengthen the attack transferability, we further introduce an audio score-matching optimization strategy to regularize the training process, which mitigates adversarial example over-fitting to the surrogate model. Extensive experiments and analysis demonstrate the effectiveness of TransAudio against open-source ASR models and commercial APIs.



## **3. Denoising Autoencoder-based Defensive Distillation as an Adversarial Robustness Algorithm**

cs.LG

This paper have 4 pages, 3 figures and it is accepted at the Ada User  journal

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15901v1) [paper-pdf](http://arxiv.org/pdf/2303.15901v1)

**Authors**: Bakary Badjie, José Cecílio, António Casimiro

**Abstract**: Adversarial attacks significantly threaten the robustness of deep neural networks (DNNs). Despite the multiple defensive methods employed, they are nevertheless vulnerable to poison attacks, where attackers meddle with the initial training data. In order to defend DNNs against such adversarial attacks, this work proposes a novel method that combines the defensive distillation mechanism with a denoising autoencoder (DAE). This technique tries to lower the sensitivity of the distilled model to poison attacks by spotting and reconstructing poisonous adversarial inputs in the training data. We added carefully created adversarial samples to the initial training data to assess the proposed method's performance. Our experimental findings demonstrate that our method successfully identified and reconstructed the poisonous inputs while also considering enhancing the DNN's resilience. The proposed approach provides a potent and robust defense mechanism for DNNs in various applications where data poisoning attacks are a concern. Thus, the defensive distillation technique's limitation posed by poisonous adversarial attacks is overcome.



## **4. Towards Effective Adversarial Textured 3D Meshes on Physical Face Recognition**

cs.CV

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15818v1) [paper-pdf](http://arxiv.org/pdf/2303.15818v1)

**Authors**: Xiao Yang, Chang Liu, Longlong Xu, Yikai Wang, Yinpeng Dong, Ning Chen, Hang Su, Jun Zhu

**Abstract**: Face recognition is a prevailing authentication solution in numerous biometric applications. Physical adversarial attacks, as an important surrogate, can identify the weaknesses of face recognition systems and evaluate their robustness before deployed. However, most existing physical attacks are either detectable readily or ineffective against commercial recognition systems. The goal of this work is to develop a more reliable technique that can carry out an end-to-end evaluation of adversarial robustness for commercial systems. It requires that this technique can simultaneously deceive black-box recognition models and evade defensive mechanisms. To fulfill this, we design adversarial textured 3D meshes (AT3D) with an elaborate topology on a human face, which can be 3D-printed and pasted on the attacker's face to evade the defenses. However, the mesh-based optimization regime calculates gradients in high-dimensional mesh space, and can be trapped into local optima with unsatisfactory transferability. To deviate from the mesh-based space, we propose to perturb the low-dimensional coefficient space based on 3D Morphable Model, which significantly improves black-box transferability meanwhile enjoying faster search efficiency and better visual quality. Extensive experiments in digital and physical scenarios show that our method effectively explores the security vulnerabilities of multiple popular commercial services, including three recognition APIs, four anti-spoofing APIs, two prevailing mobile phones and two automated access control systems.



## **5. Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization**

cs.CV

CVPR 2023

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15754v1) [paper-pdf](http://arxiv.org/pdf/2303.15754v1)

**Authors**: Jianping Zhang, Yizhan Huang, Weibin Wu, Michael R. Lyu

**Abstract**: Vision transformers (ViTs) have been successfully deployed in a variety of computer vision tasks, but they are still vulnerable to adversarial samples. Transfer-based attacks use a local model to generate adversarial samples and directly transfer them to attack a target black-box model. The high efficiency of transfer-based attacks makes it a severe security threat to ViT-based applications. Therefore, it is vital to design effective transfer-based attacks to identify the deficiencies of ViTs beforehand in security-sensitive scenarios. Existing efforts generally focus on regularizing the input gradients to stabilize the updated direction of adversarial samples. However, the variance of the back-propagated gradients in intermediate blocks of ViTs may still be large, which may make the generated adversarial samples focus on some model-specific features and get stuck in poor local optima. To overcome the shortcomings of existing approaches, we propose the Token Gradient Regularization (TGR) method. According to the structural characteristics of ViTs, TGR reduces the variance of the back-propagated gradient in each internal block of ViTs in a token-wise manner and utilizes the regularized gradient to generate adversarial samples. Extensive experiments on attacking both ViTs and CNNs confirm the superiority of our approach. Notably, compared to the state-of-the-art transfer-based attacks, our TGR offers a performance improvement of 8.8% on average.



## **6. Improving the Transferability of Adversarial Samples by Path-Augmented Method**

cs.CV

10 pages + appendix, CVPR 2023

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15735v1) [paper-pdf](http://arxiv.org/pdf/2303.15735v1)

**Authors**: Jianping Zhang, Jen-tse Huang, Wenxuan Wang, Yichen Li, Weibin Wu, Xiaosen Wang, Yuxin Su, Michael R. Lyu

**Abstract**: Deep neural networks have achieved unprecedented success on diverse vision tasks. However, they are vulnerable to adversarial noise that is imperceptible to humans. This phenomenon negatively affects their deployment in real-world scenarios, especially security-related ones. To evaluate the robustness of a target model in practice, transfer-based attacks craft adversarial samples with a local model and have attracted increasing attention from researchers due to their high efficiency. The state-of-the-art transfer-based attacks are generally based on data augmentation, which typically augments multiple training images from a linear path when learning adversarial samples. However, such methods selected the image augmentation path heuristically and may augment images that are semantics-inconsistent with the target images, which harms the transferability of the generated adversarial samples. To overcome the pitfall, we propose the Path-Augmented Method (PAM). Specifically, PAM first constructs a candidate augmentation path pool. It then settles the employed augmentation paths during adversarial sample generation with greedy search. Furthermore, to avoid augmenting semantics-inconsistent images, we train a Semantics Predictor (SP) to constrain the length of the augmentation path. Extensive experiments confirm that PAM can achieve an improvement of over 4.8% on average compared with the state-of-the-art baselines in terms of the attack success rates.



## **7. EMShepherd: Detecting Adversarial Samples via Side-channel Leakage**

cs.CR

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15571v1) [paper-pdf](http://arxiv.org/pdf/2303.15571v1)

**Authors**: Ruyi Ding, Cheng Gongye, Siyue Wang, Aidong Ding, Yunsi Fei

**Abstract**: Deep Neural Networks (DNN) are vulnerable to adversarial perturbations-small changes crafted deliberately on the input to mislead the model for wrong predictions. Adversarial attacks have disastrous consequences for deep learning-empowered critical applications. Existing defense and detection techniques both require extensive knowledge of the model, testing inputs, and even execution details. They are not viable for general deep learning implementations where the model internal is unknown, a common 'black-box' scenario for model users. Inspired by the fact that electromagnetic (EM) emanations of a model inference are dependent on both operations and data and may contain footprints of different input classes, we propose a framework, EMShepherd, to capture EM traces of model execution, perform processing on traces and exploit them for adversarial detection. Only benign samples and their EM traces are used to train the adversarial detector: a set of EM classifiers and class-specific unsupervised anomaly detectors. When the victim model system is under attack by an adversarial example, the model execution will be different from executions for the known classes, and the EM trace will be different. We demonstrate that our air-gapped EMShepherd can effectively detect different adversarial attacks on a commonly used FPGA deep learning accelerator for both Fashion MNIST and CIFAR-10 datasets. It achieves a 100% detection rate on most types of adversarial samples, which is comparable to the state-of-the-art 'white-box' software-based detectors.



## **8. Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder**

cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15564v1) [paper-pdf](http://arxiv.org/pdf/2303.15564v1)

**Authors**: Tao Sun, Lu Pang, Chao Chen, Haibin Ling

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, where an adversary maliciously manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which are impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for black-box models. The true label of every test image needs to be recovered on the fly from the hard label predictions of a suspicious model. The heuristic trigger search in image space, however, is not scalable to complex triggers or high image resolution. We circumvent such barrier by leveraging generic image generation models, and propose a framework of Blind Defense with Masked AutoEncoder (BDMAE). It uses the image structural similarity and label consistency between the test image and MAE restorations to detect possible triggers. The detection result is refined by considering the topology of triggers. We obtain a purified test image from restorations for making prediction. Our approach is blind to the model architectures, trigger patterns or image benignity. Extensive experiments on multiple datasets with different backdoor attacks validate its effectiveness and generalizability. Code is available at https://github.com/tsun/BDMAE.



## **9. Intel TDX Demystified: A Top-Down Approach**

cs.CR

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15540v1) [paper-pdf](http://arxiv.org/pdf/2303.15540v1)

**Authors**: Pau-Chen Cheng, Wojciech Ozga, Enriquillo Valdez, Salman Ahmed, Zhongshu Gu, Hani Jamjoom, Hubertus Franke, James Bottomley

**Abstract**: Intel Trust Domain Extensions (TDX) is a new architectural extension in the 4th Generation Intel Xeon Scalable Processor that supports confidential computing. TDX allows the deployment of virtual machines in the Secure-Arbitration Mode (SEAM) with encrypted CPU state and memory, integrity protection, and remote attestation. TDX aims to enforce hardware-assisted isolation for virtual machines and minimize the attack surface exposed to host platforms, which are considered to be untrustworthy or adversarial in the confidential computing's new threat model. TDX can be leveraged by regulated industries or sensitive data holders to outsource their computations and data with end-to-end protection in public cloud infrastructure.   This paper aims to provide a comprehensive understanding of TDX to potential adopters, domain experts, and security researchers looking to leverage the technology for their own purposes. We adopt a top-down approach, starting with high-level security principles and moving to low-level technical details of TDX. Our analysis is based on publicly available documentation and source code, offering insights from security researchers outside of Intel.



## **10. Classifier Robustness Enhancement Via Test-Time Transformation**

cs.CV

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15409v1) [paper-pdf](http://arxiv.org/pdf/2303.15409v1)

**Authors**: Tsachi Blau, Roy Ganz, Chaim Baskin, Michael Elad, Alex Bronstein

**Abstract**: It has been recently discovered that adversarially trained classifiers exhibit an intriguing property, referred to as perceptually aligned gradients (PAG). PAG implies that the gradients of such classifiers possess a meaningful structure, aligned with human perception. Adversarial training is currently the best-known way to achieve classification robustness under adversarial attacks. The PAG property, however, has yet to be leveraged for further improving classifier robustness. In this work, we introduce Classifier Robustness Enhancement Via Test-Time Transformation (TETRA) -- a novel defense method that utilizes PAG, enhancing the performance of trained robust classifiers. Our method operates in two phases. First, it modifies the input image via a designated targeted adversarial attack into each of the dataset's classes. Then, it classifies the input image based on the distance to each of the modified instances, with the assumption that the shortest distance relates to the true class. We show that the proposed method achieves state-of-the-art results and validate our claim through extensive experiments on a variety of defense methods, classifier architectures, and datasets. We also empirically demonstrate that TETRA can boost the accuracy of any differentiable adversarial training classifier across a variety of attacks, including ones unseen at training. Specifically, applying TETRA leads to substantial improvement of up to $+23\%$, $+20\%$, and $+26\%$ on CIFAR10, CIFAR100, and ImageNet, respectively.



## **11. Learning the Unlearnable: Adversarial Augmentations Suppress Unlearnable Example Attacks**

cs.LG

UEraser introduces adversarial augmentations to suppress unlearnable  example attacks and outperforms current defenses

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15127v1) [paper-pdf](http://arxiv.org/pdf/2303.15127v1)

**Authors**: Tianrui Qin, Xitong Gao, Juanjuan Zhao, Kejiang Ye, Cheng-Zhong Xu

**Abstract**: Unlearnable example attacks are data poisoning techniques that can be used to safeguard public data against unauthorized use for training deep learning models. These methods add stealthy perturbations to the original image, thereby making it difficult for deep learning models to learn from these training data effectively. Current research suggests that adversarial training can, to a certain degree, mitigate the impact of unlearnable example attacks, while common data augmentation methods are not effective against such poisons. Adversarial training, however, demands considerable computational resources and can result in non-trivial accuracy loss. In this paper, we introduce the UEraser method, which outperforms current defenses against different types of state-of-the-art unlearnable example attacks through a combination of effective data augmentation policies and loss-maximizing adversarial augmentations. In stark contrast to the current SOTA adversarial training methods, UEraser uses adversarial augmentations, which extends beyond the confines of $ \ell_p $ perturbation budget assumed by current unlearning attacks and defenses. It also helps to improve the model's generalization ability, thus protecting against accuracy loss. UEraser wipes out the unlearning effect with error-maximizing data augmentations, thus restoring trained model accuracies. Interestingly, UEraser-Lite, a fast variant without adversarial augmentations, is also highly effective in preserving clean accuracies. On challenging unlearnable CIFAR-10, CIFAR-100, SVHN, and ImageNet-subset datasets produced with various attacks, it achieves results that are comparable to those obtained during clean training. We also demonstrate its efficacy against possible adaptive attacks. Our code is open source and available to the deep learning community: https://github.com/lafeat/ueraser.



## **12. Among Us: Adversarially Robust Collaborative Perception by Consensus**

cs.RO

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.09495v2) [paper-pdf](http://arxiv.org/pdf/2303.09495v2)

**Authors**: Yiming Li, Qi Fang, Jiamu Bai, Siheng Chen, Felix Juefei-Xu, Chen Feng

**Abstract**: Multiple robots could perceive a scene (e.g., detect objects) collaboratively better than individuals, although easily suffer from adversarial attacks when using deep learning. This could be addressed by the adversarial defense, but its training requires the often-unknown attacking mechanism. Differently, we propose ROBOSAC, a novel sampling-based defense strategy generalizable to unseen attackers. Our key idea is that collaborative perception should lead to consensus rather than dissensus in results compared to individual perception. This leads to our hypothesize-and-verify framework: perception results with and without collaboration from a random subset of teammates are compared until reaching a consensus. In such a framework, more teammates in the sampled subset often entail better perception performance but require longer sampling time to reject potential attackers. Thus, we derive how many sampling trials are needed to ensure the desired size of an attacker-free subset, or equivalently, the maximum size of such a subset that we can successfully sample within a given number of trials. We validate our method on the task of collaborative 3D object detection in autonomous driving scenarios.



## **13. Identifying Adversarially Attackable and Robust Samples**

cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2301.12896v2) [paper-pdf](http://arxiv.org/pdf/2301.12896v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Adversarial attacks insert small, imperceptible perturbations to input samples that cause large, undesired changes to the output of deep learning models. Despite extensive research on generating adversarial attacks and building defense systems, there has been limited research on understanding adversarial attacks from an input-data perspective. This work introduces the notion of sample attackability, where we aim to identify samples that are most susceptible to adversarial attacks (attackable samples) and conversely also identify the least susceptible samples (robust samples). We propose a deep-learning-based method to detect the adversarially attackable and robust samples in an unseen dataset for an unseen target model. Experiments on standard image classification datasets enables us to assess the portability of the deep attackability detector across a range of architectures. We find that the deep attackability detector performs better than simple model uncertainty-based measures for identifying the attackable/robust samples. This suggests that uncertainty is an inadequate proxy for measuring sample distance to a decision boundary. In addition to better understanding adversarial attack theory, it is found that the ability to identify the adversarially attackable and robust samples has implications for improving the efficiency of sample-selection tasks, e.g. active learning in augmentation for adversarial training.



## **14. Improving the Transferability of Adversarial Examples via Direction Tuning**

cs.CV

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15109v1) [paper-pdf](http://arxiv.org/pdf/2303.15109v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstract**: In the transfer-based adversarial attacks, adversarial examples are only generated by the surrogate models and achieve effective perturbation in the victim models. Although considerable efforts have been developed on improving the transferability of adversarial examples generated by transfer-based adversarial attacks, our investigation found that, the big deviation between the actual and steepest update directions of the current transfer-based adversarial attacks is caused by the large update step length, resulting in the generated adversarial examples can not converge well. However, directly reducing the update step length will lead to serious update oscillation so that the generated adversarial examples also can not achieve great transferability to the victim models. To address these issues, a novel transfer-based attack, namely direction tuning attack, is proposed to not only decrease the update deviation in the large step length, but also mitigate the update oscillation in the small sampling step length, thereby making the generated adversarial examples converge well to achieve great transferability on victim models. In addition, a network pruning method is proposed to smooth the decision boundary, thereby further decreasing the update oscillation and enhancing the transferability of the generated adversarial examples. The experiment results on ImageNet demonstrate that the average attack success rate (ASR) of the adversarial examples generated by our method can be improved from 87.9\% to 94.5\% on five victim models without defenses, and from 69.1\% to 76.2\% on eight advanced defense methods, in comparison with that of latest gradient-based attacks.



## **15. Improved Adversarial Training Through Adaptive Instance-wise Loss Smoothing**

cs.CV

12 pages, work in submission

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14077v2) [paper-pdf](http://arxiv.org/pdf/2303.14077v2)

**Authors**: Lin Li, Michael Spratling

**Abstract**: Deep neural networks can be easily fooled into making incorrect predictions through corruption of the input by adversarial perturbations: human-imperceptible artificial noise. So far adversarial training has been the most successful defense against such adversarial attacks. This work focuses on improving adversarial training to boost adversarial robustness. We first analyze, from an instance-wise perspective, how adversarial vulnerability evolves during adversarial training. We find that during training an overall reduction of adversarial loss is achieved by sacrificing a considerable proportion of training samples to be more vulnerable to adversarial attack, which results in an uneven distribution of adversarial vulnerability among data. Such "uneven vulnerability", is prevalent across several popular robust training methods and, more importantly, relates to overfitting in adversarial training. Motivated by this observation, we propose a new adversarial training method: Instance-adaptive Smoothness Enhanced Adversarial Training (ISEAT). It jointly smooths both input and weight loss landscapes in an adaptive, instance-specific, way to enhance robustness more for those samples with higher adversarial vulnerability. Extensive experiments demonstrate the superiority of our method over existing defense methods. Noticeably, our method, when combined with the latest data augmentation and semi-supervised learning techniques, achieves state-of-the-art robustness against $\ell_{\infty}$-norm constrained attacks on CIFAR10 of 59.32% for Wide ResNet34-10 without extra data, and 61.55% for Wide ResNet28-10 with extra data. Code is available at https://github.com/TreeLLi/Instance-adaptive-Smoothness-Enhanced-AT.



## **16. Diffusion Denoised Smoothing for Certified and Adversarial Robust Out-Of-Distribution Detection**

cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14961v1) [paper-pdf](http://arxiv.org/pdf/2303.14961v1)

**Authors**: Nicola Franco, Daniel Korth, Jeanette Miriam Lorenz, Karsten Roscher, Stephan Guennemann

**Abstract**: As the use of machine learning continues to expand, the importance of ensuring its safety cannot be overstated. A key concern in this regard is the ability to identify whether a given sample is from the training distribution, or is an "Out-Of-Distribution" (OOD) sample. In addition, adversaries can manipulate OOD samples in ways that lead a classifier to make a confident prediction. In this study, we present a novel approach for certifying the robustness of OOD detection within a $\ell_2$-norm around the input, regardless of network architecture and without the need for specific components or additional training. Further, we improve current techniques for detecting adversarial attacks on OOD samples, while providing high levels of certified and adversarial robustness on in-distribution samples. The average of all OOD detection metrics on CIFAR10/100 shows an increase of $\sim 13 \% / 5\%$ relative to previous approaches.



## **17. CAT:Collaborative Adversarial Training**

cs.CV

Tech report

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14922v1) [paper-pdf](http://arxiv.org/pdf/2303.14922v1)

**Authors**: Xingbin Liu, Huafeng Kuang, Xianming Lin, Yongjian Wu, Rongrong Ji

**Abstract**: Adversarial training can improve the robustness of neural networks. Previous methods focus on a single adversarial training strategy and do not consider the model property trained by different strategies. By revisiting the previous methods, we find different adversarial training methods have distinct robustness for sample instances. For example, a sample instance can be correctly classified by a model trained using standard adversarial training (AT) but not by a model trained using TRADES, and vice versa. Based on this observation, we propose a collaborative adversarial training framework to improve the robustness of neural networks. Specifically, we use different adversarial training methods to train robust models and let models interact with their knowledge during the training process. Collaborative Adversarial Training (CAT) can improve both robustness and accuracy. Extensive experiments on various networks and datasets validate the effectiveness of our method. CAT achieves state-of-the-art adversarial robustness without using any additional data on CIFAR-10 under the Auto-Attack benchmark. Code is available at https://github.com/liuxingbin/CAT.



## **18. Efficient Robustness Assessment via Adversarial Spatial-Temporal Focus on Videos**

cs.CV

accepted by TPAMI2023

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2301.00896v2) [paper-pdf](http://arxiv.org/pdf/2301.00896v2)

**Authors**: Wei Xingxing, Wang Songping, Yan Huanqian

**Abstract**: Adversarial robustness assessment for video recognition models has raised concerns owing to their wide applications on safety-critical tasks. Compared with images, videos have much high dimension, which brings huge computational costs when generating adversarial videos. This is especially serious for the query-based black-box attacks where gradient estimation for the threat models is usually utilized, and high dimensions will lead to a large number of queries. To mitigate this issue, we propose to simultaneously eliminate the temporal and spatial redundancy within the video to achieve an effective and efficient gradient estimation on the reduced searching space, and thus query number could decrease. To implement this idea, we design the novel Adversarial spatial-temporal Focus (AstFocus) attack on videos, which performs attacks on the simultaneously focused key frames and key regions from the inter-frames and intra-frames in the video. AstFocus attack is based on the cooperative Multi-Agent Reinforcement Learning (MARL) framework. One agent is responsible for selecting key frames, and another agent is responsible for selecting key regions. These two agents are jointly trained by the common rewards received from the black-box threat models to perform a cooperative prediction. By continuously querying, the reduced searching space composed of key frames and key regions is becoming precise, and the whole query number becomes less than that on the original video. Extensive experiments on four mainstream video recognition models and three widely used action recognition datasets demonstrate that the proposed AstFocus attack outperforms the SOTA methods, which is prevenient in fooling rate, query number, time, and perturbation magnitude at the same.



## **19. Don't be a Victim During a Pandemic! Analysing Security and Privacy Threats in Twitter During COVID-19**

cs.CR

Paper has been accepted for publication in IEEE Access. Currently  available on IEEE ACCESS early access (see DOI)

**SubmitDate**: 2023-03-26    [abs](http://arxiv.org/abs/2202.10543v2) [paper-pdf](http://arxiv.org/pdf/2202.10543v2)

**Authors**: Bibhas Sharma, Ishan Karunanayake, Rahat Masood, Muhammad Ikram

**Abstract**: There has been a huge spike in the usage of social media platforms during the COVID-19 lockdowns. These lockdown periods have resulted in a set of new cybercrimes, thereby allowing attackers to victimise social media users with a range of threats. This paper performs a large-scale study to investigate the impact of a pandemic and the lockdown periods on the security and privacy of social media users. We analyse 10.6 Million COVID-related tweets from 533 days of data crawling and investigate users' security and privacy behaviour in three different periods (i.e., before, during, and after the lockdown). Our study shows that users unintentionally share more personal identifiable information when writing about the pandemic situation (e.g., sharing nearby coronavirus testing locations) in their tweets. The privacy risk reaches 100% if a user posts three or more sensitive tweets about the pandemic. We investigate the number of suspicious domains shared on social media during different phases of the pandemic. Our analysis reveals an increase in the number of suspicious domains during the lockdown compared to other lockdown phases. We observe that IT, Search Engines, and Businesses are the top three categories that contain suspicious domains. Our analysis reveals that adversaries' strategies to instigate malicious activities change with the country's pandemic situation.



## **20. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

stat.ML

9 pages; Previously this version appeared as arXiv:2210.08198 which  was submitted as a new work by accident

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2109.12772v2) [paper-pdf](http://arxiv.org/pdf/2109.12772v2)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.



## **21. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

cs.CV

This work was intended as a replacement of arXiv:2109.12772 and any  subsequent updates will appear there

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2210.08198v2) [paper-pdf](http://arxiv.org/pdf/2210.08198v2)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Ch. Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.



## **22. STDLens: Model Hijacking-Resilient Federated Learning for Object Detection**

cs.CR

CVPR 2023. Source Code: https://github.com/git-disl/STDLens

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.11511v2) [paper-pdf](http://arxiv.org/pdf/2303.11511v2)

**Authors**: Ka-Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu

**Abstract**: Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates.



## **23. Improving robustness of jet tagging algorithms with adversarial training: exploring the loss surface**

hep-ex

5 pages, 2 figures; submitted to ACAT 2022 proceedings

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14511v1) [paper-pdf](http://arxiv.org/pdf/2303.14511v1)

**Authors**: Annika Stein

**Abstract**: In the field of high-energy physics, deep learning algorithms continue to gain in relevance and provide performance improvements over traditional methods, for example when identifying rare signals or finding complex patterns. From an analyst's perspective, obtaining highest possible performance is desirable, but recently, some attention has been shifted towards studying robustness of models to investigate how well these perform under slight distortions of input features. Especially for tasks that involve many (low-level) inputs, the application of deep neural networks brings new challenges. In the context of jet flavor tagging, adversarial attacks are used to probe a typical classifier's vulnerability and can be understood as a model for systematic uncertainties. A corresponding defense strategy, adversarial training, improves robustness, while maintaining high performance. Investigating the loss surface corresponding to the inputs and models in question reveals geometric interpretations of robustness, taking correlations into account.



## **24. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

cs.CV

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2204.10779v6) [paper-pdf](http://arxiv.org/pdf/2204.10779v6)

**Authors**: Xunguang Wang, Yiqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61\%, 12.35\%, and 11.56\% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively. The code is available at https://github.com/xunguangwang/CgAT.



## **25. No more Reviewer #2: Subverting Automatic Paper-Reviewer Assignment using Adversarial Learning**

cs.CR

Accepted at USENIX Security Symposium 2023

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14443v1) [paper-pdf](http://arxiv.org/pdf/2303.14443v1)

**Authors**: Thorsten Eisenhofer, Erwin Quiring, Jonas Möller, Doreen Riepel, Thorsten Holz, Konrad Rieck

**Abstract**: The number of papers submitted to academic conferences is steadily rising in many scientific disciplines. To handle this growth, systems for automatic paper-reviewer assignments are increasingly used during the reviewing process. These systems use statistical topic models to characterize the content of submissions and automate the assignment to reviewers. In this paper, we show that this automation can be manipulated using adversarial learning. We propose an attack that adapts a given paper so that it misleads the assignment and selects its own reviewers. Our attack is based on a novel optimization strategy that alternates between the feature space and problem space to realize unobtrusive changes to the paper. To evaluate the feasibility of our attack, we simulate the paper-reviewer assignment of an actual security conference (IEEE S&P) with 165 reviewers on the program committee. Our results show that we can successfully select and remove reviewers without access to the assignment system. Moreover, we demonstrate that the manipulated papers remain plausible and are often indistinguishable from benign submissions.



## **26. A User-Based Authentication and DoS Mitigation Scheme for Wearable Wireless Body Sensor Networks**

cs.CR

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14441v1) [paper-pdf](http://arxiv.org/pdf/2303.14441v1)

**Authors**: Nombulelo Zulu, Deon P. Du Plessis, Topside E. Mathonsi, Tshimangadzo M. Tshilongamulenzhe

**Abstract**: Wireless Body Sensor Networks (WBSNs) is one of the greatest growing technology for sensing and performing various tasks. The information transmitted in the WBSNs is vulnerable to cyber-attacks, therefore security is very important. Denial of Service (DoS) attacks are considered one of the major threats against WBSNs security. In DoS attacks, an adversary targets to degrade and shut down the efficient use of the network and disrupt the services in the network causing them inaccessible to its intended users. If sensitive information of patients in WBSNs, such as the medical history is accessed by unauthorized users, the patient may suffer much more than the disease itself, it may result in loss of life. This paper proposes a User-Based authentication scheme to mitigate DoS attacks in WBSNs. A five-phase User-Based authentication DoS mitigation scheme for WBSNs is designed by integrating Elliptic Curve Cryptography (ECC) with Rivest Cipher 4 (RC4) to ensure a strong authentication process that will only allow authorized users to access nodes on WBSNs.



## **27. Consistent Attack: Universal Adversarial Perturbation on Embodied Vision Navigation**

cs.LG

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2206.05751v4) [paper-pdf](http://arxiv.org/pdf/2206.05751v4)

**Authors**: Chengyang Ying, You Qiaoben, Xinning Zhou, Hang Su, Wenbo Ding, Jianyong Ai

**Abstract**: Embodied agents in vision navigation coupled with deep neural networks have attracted increasing attention. However, deep neural networks have been shown vulnerable to malicious adversarial noises, which may potentially cause catastrophic failures in Embodied Vision Navigation. Among different adversarial noises, universal adversarial perturbations (UAP), i.e., a constant image-agnostic perturbation applied on every input frame of the agent, play a critical role in Embodied Vision Navigation since they are computation-efficient and application-practical during the attack. However, existing UAP methods ignore the system dynamics of Embodied Vision Navigation and might be sub-optimal. In order to extend UAP to the sequential decision setting, we formulate the disturbed environment under the universal noise $\delta$, as a $\delta$-disturbed Markov Decision Process ($\delta$-MDP). Based on the formulation, we analyze the properties of $\delta$-MDP and propose two novel Consistent Attack methods, named Reward UAP and Trajectory UAP, for attacking Embodied agents, which consider the dynamic of the MDP and calculate universal noises by estimating the disturbed distribution and the disturbed Q function. For various victim models, our Consistent Attack can cause a significant drop in their performance in the PointGoal task in Habitat with different datasets and different scenes. Extensive experimental results indicate that there exist serious potential risks for applying Embodied Vision Navigation methods to the real world.



## **28. Test-time Defense against Adversarial Attacks: Detection and Reconstruction of Adversarial Examples via Masked Autoencoder**

cs.CV

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.12848v2) [paper-pdf](http://arxiv.org/pdf/2303.12848v2)

**Authors**: Yun-Yun Tsai, Ju-Chin Chao, Albert Wen, Zhaoyuan Yang, Chengzhi Mao, Tapan Shah, Junfeng Yang

**Abstract**: Existing defense methods against adversarial attacks can be categorized into training time and test time defenses. Training time defense, i.e., adversarial training, requires a significant amount of extra time for training and is often not able to be generalized to unseen attacks. On the other hand, test time defense by test time weight adaptation requires access to perform gradient descent on (part of) the model weights, which could be infeasible for models with frozen weights. To address these challenges, we propose DRAM, a novel defense method to Detect and Reconstruct multiple types of Adversarial attacks via Masked autoencoder (MAE). We demonstrate how to use MAE losses to build a KS-test to detect adversarial attacks. Moreover, the MAE losses can be used to repair adversarial samples from unseen attack types. In this sense, DRAM neither requires model weight updates in test time nor augments the training set with more adversarial samples. Evaluating DRAM on the large-scale ImageNet data, we achieve the best detection rate of 82% on average on eight types of adversarial attacks compared with other detection baselines. For reconstruction, DRAM improves the robust accuracy by 6% ~ 41% for Standard ResNet50 and 3% ~ 8% for Robust ResNet50 compared with other self-supervision tasks, such as rotation prediction and contrastive learning.



## **29. WiFi Physical Layer Stays Awake and Responds When it Should Not**

cs.NI

12 pages

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2301.00269v2) [paper-pdf](http://arxiv.org/pdf/2301.00269v2)

**Authors**: Ali Abedi, Haofan Lu, Alex Chen, Charlie Liu, Omid Abari

**Abstract**: WiFi communication should be possible only between devices inside the same network. However, we find that all existing WiFi devices send back acknowledgments (ACK) to even fake packets received from unauthorized WiFi devices outside of their network. Moreover, we find that an unauthorized device can manipulate the power-saving mechanism of WiFi radios and keep them continuously awake by sending specific fake beacon frames to them. Our evaluation of over 5,000 devices from 186 vendors confirms that these are widespread issues. We believe these loopholes cannot be prevented, and hence they create privacy and security concerns. Finally, to show the importance of these issues and their consequences, we implement and demonstrate two attacks where an adversary performs battery drain and WiFi sensing attacks just using a tiny WiFi module which costs less than ten dollars.



## **30. Ensemble-based Blackbox Attacks on Dense Prediction**

cs.CV

CVPR 2023 Accepted

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14304v1) [paper-pdf](http://arxiv.org/pdf/2303.14304v1)

**Authors**: Zikui Cai, Yaoteng Tan, M. Salman Asif

**Abstract**: We propose an approach for adversarial attacks on dense prediction models (such as object detectors and segmentation). It is well known that the attacks generated by a single surrogate model do not transfer to arbitrary (blackbox) victim models. Furthermore, targeted attacks are often more challenging than the untargeted attacks. In this paper, we show that a carefully designed ensemble can create effective attacks for a number of victim models. In particular, we show that normalization of the weights for individual models plays a critical role in the success of the attacks. We then demonstrate that by adjusting the weights of the ensemble according to the victim model can further improve the performance of the attacks. We performed a number of experiments for object detectors and segmentation to highlight the significance of the our proposed methods. Our proposed ensemble-based method outperforms existing blackbox attack methods for object detection and segmentation. Finally we show that our proposed method can also generate a single perturbation that can fool multiple blackbox detection and segmentation models simultaneously. Code is available at https://github.com/CSIPlab/EBAD.



## **31. Utilizing Network Properties to Detect Erroneous Inputs**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2002.12520v3) [paper-pdf](http://arxiv.org/pdf/2002.12520v3)

**Authors**: Matt Gorbett, Nathaniel Blanchard

**Abstract**: Neural networks are vulnerable to a wide range of erroneous inputs such as adversarial, corrupted, out-of-distribution, and misclassified examples. In this work, we train a linear SVM classifier to detect these four types of erroneous data using hidden and softmax feature vectors of pre-trained neural networks. Our results indicate that these faulty data types generally exhibit linearly separable activation properties from correct examples, giving us the ability to reject bad inputs with no extra training or overhead. We experimentally validate our findings across a diverse range of datasets, domains, pre-trained models, and adversarial attacks.



## **32. How many dimensions are required to find an adversarial example?**

cs.LG

Comments welcome!

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14173v1) [paper-pdf](http://arxiv.org/pdf/2303.14173v1)

**Authors**: Charles Godfrey, Henry Kvinge, Elise Bishoff, Myles Mckay, Davis Brown, Tim Doster, Eleanor Byler

**Abstract**: Past work exploring adversarial vulnerability have focused on situations where an adversary can perturb all dimensions of model input. On the other hand, a range of recent works consider the case where either (i) an adversary can perturb a limited number of input parameters or (ii) a subset of modalities in a multimodal problem. In both of these cases, adversarial examples are effectively constrained to a subspace $V$ in the ambient input space $\mathcal{X}$. Motivated by this, in this work we investigate how adversarial vulnerability depends on $\dim(V)$. In particular, we show that the adversarial success of standard PGD attacks with $\ell^p$ norm constraints behaves like a monotonically increasing function of $\epsilon (\frac{\dim(V)}{\dim \mathcal{X}})^{\frac{1}{q}}$ where $\epsilon$ is the perturbation budget and $\frac{1}{p} + \frac{1}{q} =1$, provided $p > 1$ (the case $p=1$ presents additional subtleties which we analyze in some detail). This functional form can be easily derived from a simple toy linear model, and as such our results land further credence to arguments that adversarial examples are endemic to locally linear models on high dimensional spaces.



## **33. Adversarial Attack and Defense for Medical Image Analysis: Methods and Applications**

eess.IV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14133v1) [paper-pdf](http://arxiv.org/pdf/2303.14133v1)

**Authors**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen

**Abstract**: Deep learning techniques have achieved superior performance in computer-aided medical image analysis, yet they are still vulnerable to imperceptible adversarial attacks, resulting in potential misdiagnosis in clinical practice. Oppositely, recent years have also witnessed remarkable progress in defense against these tailored adversarial examples in deep medical diagnosis systems. In this exposition, we present a comprehensive survey on recent advances in adversarial attack and defense for medical image analysis with a novel taxonomy in terms of the application scenario. We also provide a unified theoretical framework for different types of adversarial attack and defense methods for medical image analysis. For a fair comparison, we establish a new benchmark for adversarially robust medical diagnosis models obtained by adversarial training under various scenarios. To the best of our knowledge, this is the first survey paper that provides a thorough evaluation of adversarially robust medical diagnosis models. By analyzing qualitative and quantitative results, we conclude this survey with a detailed discussion of current challenges for adversarial attack and defense in medical image analysis systems to shed light on future research directions.



## **34. Optimal Smoothing Distribution Exploration for Backdoor Neutralization in Deep Learning-based Traffic Systems**

cs.LG

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14197v1) [paper-pdf](http://arxiv.org/pdf/2303.14197v1)

**Authors**: Yue Wang, Wending Li, Michail Maniatakos, Saif Eddin Jabari

**Abstract**: Deep Reinforcement Learning (DRL) enhances the efficiency of Autonomous Vehicles (AV), but also makes them susceptible to backdoor attacks that can result in traffic congestion or collisions. Backdoor functionality is typically incorporated by contaminating training datasets with covert malicious data to maintain high precision on genuine inputs while inducing the desired (malicious) outputs for specific inputs chosen by adversaries. Current defenses against backdoors mainly focus on image classification using image-based features, which cannot be readily transferred to the regression task of DRL-based AV controllers since the inputs are continuous sensor data, i.e., the combinations of velocity and distance of AV and its surrounding vehicles. Our proposed method adds well-designed noise to the input to neutralize backdoors. The approach involves learning an optimal smoothing (noise) distribution to preserve the normal functionality of genuine inputs while neutralizing backdoors. By doing so, the resulting model is expected to be more resilient against backdoor attacks while maintaining high accuracy on genuine inputs. The effectiveness of the proposed method is verified on a simulated traffic system based on a microscopic traffic simulator, where experimental results showcase that the smoothed traffic controller can neutralize all trigger samples and maintain the performance of relieving traffic congestion



## **35. PIAT: Parameter Interpolation based Adversarial Training for Image Classification**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13955v1) [paper-pdf](http://arxiv.org/pdf/2303.13955v1)

**Authors**: Kun He, Xin Liu, Yichen Yang, Zhou Qin, Weigao Wen, Hui Xue, John E. Hopcroft

**Abstract**: Adversarial training has been demonstrated to be the most effective approach to defend against adversarial attacks. However, existing adversarial training methods show apparent oscillations and overfitting issue in the training process, degrading the defense efficacy. In this work, we propose a novel framework, termed Parameter Interpolation based Adversarial Training (PIAT), that makes full use of the historical information during training. Specifically, at the end of each epoch, PIAT tunes the model parameters as the interpolation of the parameters of the previous and current epochs. Besides, we suggest to use the Normalized Mean Square Error (NMSE) to further improve the robustness by aligning the clean and adversarial examples. Compared with other regularization methods, NMSE focuses more on the relative magnitude of the logits rather than the absolute magnitude. Extensive experiments on several benchmark datasets and various networks show that our method could prominently improve the model robustness and reduce the generalization error. Moreover, our framework is general and could further boost the robust accuracy when combined with other adversarial training methods.



## **36. EC-CFI: Control-Flow Integrity via Code Encryption Counteracting Fault Attacks**

cs.CR

Accepted at HOST'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2301.13760v2) [paper-pdf](http://arxiv.org/pdf/2301.13760v2)

**Authors**: Pascal Nasahl, Salmin Sultana, Hans Liljestrand, Karanvir Grewal, Michael LeMay, David M. Durham, David Schrammel, Stefan Mangard

**Abstract**: Fault attacks enable adversaries to manipulate the control-flow of security-critical applications. By inducing targeted faults into the CPU, the software's call graph can be escaped and the control-flow can be redirected to arbitrary functions inside the program. To protect the control-flow from these attacks, dedicated fault control-flow integrity (CFI) countermeasures are commonly deployed. However, these schemes either have high detection latencies or require intrusive hardware changes. In this paper, we present EC-CFI, a software-based cryptographically enforced CFI scheme with no detection latency utilizing hardware features of recent Intel platforms. Our EC-CFI prototype is designed to prevent an adversary from escaping the program's call graph using faults by encrypting each function with a different key before execution. At runtime, the instrumented program dynamically derives the decryption key, ensuring that the code only can be successfully decrypted when the program follows the intended call graph. To enable this level of protection on Intel commodity systems, we introduce extended page table (EPT) aliasing allowing us to achieve function-granular encryption by combing Intel's TME-MK and virtualization technology. We open-source our custom LLVM-based toolchain automatically protecting arbitrary programs with EC-CFI. Furthermore, we evaluate our EPT aliasing approach with the SPEC CPU2017 and Embench-IoT benchmarks and discuss and evaluate potential TME-MK hardware changes minimizing runtime overheads.



## **37. SCRAMBLE-CFI: Mitigating Fault-Induced Control-Flow Attacks on OpenTitan**

cs.CR

Accepted at GLSVLSI'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.03711v3) [paper-pdf](http://arxiv.org/pdf/2303.03711v3)

**Authors**: Pascal Nasahl, Stefan Mangard

**Abstract**: Secure elements physically exposed to adversaries are frequently targeted by fault attacks. These attacks can be utilized to hijack the control-flow of software allowing the attacker to bypass security measures, extract sensitive data, or gain full code execution. In this paper, we systematically analyze the threat vector of fault-induced control-flow manipulations on the open-source OpenTitan secure element. Our thorough analysis reveals that current countermeasures of this chip either induce large area overheads or still cannot prevent the attacker from exploiting the identified threats. In this context, we introduce SCRAMBLE-CFI, an encryption-based control-flow integrity scheme utilizing existing hardware features of OpenTitan. SCRAMBLE-CFI confines, with minimal hardware overhead, the impact of fault-induced control-flow attacks by encrypting each function with a different encryption tweak at load-time. At runtime, code only can be successfully decrypted when the correct decryption tweak is active. We open-source our hardware changes and release our LLVM toolchain automatically protecting programs. Our analysis shows that SCRAMBLE-CFI complementarily enhances security guarantees of OpenTitan with a negligible hardware overhead of less than 3.97 % and a runtime overhead of 7.02 % for the Embench-IoT benchmarks.



## **38. Foiling Explanations in Deep Neural Networks**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2211.14860v2) [paper-pdf](http://arxiv.org/pdf/2211.14860v2)

**Authors**: Snir Vitrack Tamam, Raz Lapid, Moshe Sipper

**Abstract**: Deep neural networks (DNNs) have greatly impacted numerous fields over the past decade. Yet despite exhibiting superb performance over many problems, their black-box nature still poses a significant challenge with respect to explainability. Indeed, explainable artificial intelligence (XAI) is crucial in several fields, wherein the answer alone -- sans a reasoning of how said answer was derived -- is of little value. This paper uncovers a troubling property of explanation methods for image-based DNNs: by making small visual changes to the input image -- hardly influencing the network's output -- we demonstrate how explanations may be arbitrarily manipulated through the use of evolution strategies. Our novel algorithm, AttaXAI, a model-agnostic, adversarial attack on XAI algorithms, only requires access to the output logits of a classifier and to the explanation map; these weak assumptions render our approach highly useful where real-world models and data are concerned. We compare our method's performance on two benchmark datasets -- CIFAR100 and ImageNet -- using four different pretrained deep-learning models: VGG16-CIFAR100, VGG16-ImageNet, MobileNet-CIFAR100, and Inception-v3-ImageNet. We find that the XAI methods can be manipulated without the use of gradients or other model internals. Our novel algorithm is successfully able to manipulate an image in a manner imperceptible to the human eye, such that the XAI method outputs a specific explanation map. To our knowledge, this is the first such method in a black-box setting, and we believe it has significant value where explainability is desired, required, or legally mandatory.



## **39. Effective black box adversarial attack with handcrafted kernels**

cs.CV

12 pages, 5 figures, 3 tables, IWANN conference

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13887v1) [paper-pdf](http://arxiv.org/pdf/2303.13887v1)

**Authors**: Petr Dvořáček, Petr Hurtik, Petra Števuliáková

**Abstract**: We propose a new, simple framework for crafting adversarial examples for black box attacks. The idea is to simulate the substitution model with a non-trainable model compounded of just one layer of handcrafted convolutional kernels and then train the generator neural network to maximize the distance of the outputs for the original and generated adversarial image. We show that fooling the prediction of the first layer causes the whole network to be fooled and decreases its accuracy on adversarial inputs. Moreover, we do not train the neural network to obtain the first convolutional layer kernels, but we create them using the technique of F-transform. Therefore, our method is very time and resource effective.



## **40. Physically Adversarial Infrared Patches with Learnable Shapes and Locations**

cs.CV

accepted by CVPR2023

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13868v1) [paper-pdf](http://arxiv.org/pdf/2303.13868v1)

**Authors**: Wei Xingxing, Yu Jie, Huang Yao

**Abstract**: Owing to the extensive application of infrared object detectors in the safety-critical tasks, it is necessary to evaluate their robustness against adversarial examples in the real world. However, current few physical infrared attacks are complicated to implement in practical application because of their complex transformation from digital world to physical world. To address this issue, in this paper, we propose a physically feasible infrared attack method called "adversarial infrared patches". Considering the imaging mechanism of infrared cameras by capturing objects' thermal radiation, adversarial infrared patches conduct attacks by attaching a patch of thermal insulation materials on the target object to manipulate its thermal distribution. To enhance adversarial attacks, we present a novel aggregation regularization to guide the simultaneous learning for the patch' shape and location on the target object. Thus, a simple gradient-based optimization can be adapted to solve for them. We verify adversarial infrared patches in different object detection tasks with various object detectors. Experimental results show that our method achieves more than 90\% Attack Success Rate (ASR) versus the pedestrian detector and vehicle detector in the physical environment, where the objects are captured in different angles, distances, postures, and scenes. More importantly, adversarial infrared patch is easy to implement, and it only needs 0.5 hours to be constructed in the physical world, which verifies its effectiveness and efficiency.



## **41. Feature Separation and Recalibration for Adversarial Robustness**

cs.CV

CVPR 2023 (Highlight)

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13846v1) [paper-pdf](http://arxiv.org/pdf/2303.13846v1)

**Authors**: Woo Jae Kim, Yoonki Cho, Junsik Jung, Sung-Eui Yoon

**Abstract**: Deep neural networks are susceptible to adversarial attacks due to the accumulation of perturbations in the feature level, and numerous works have boosted model robustness by deactivating the non-robust feature activations that cause model mispredictions. However, we claim that these malicious activations still contain discriminative cues and that with recalibration, they can capture additional useful information for correct model predictions. To this end, we propose a novel, easy-to-plugin approach named Feature Separation and Recalibration (FSR) that recalibrates the malicious, non-robust activations for more robust feature maps through Separation and Recalibration. The Separation part disentangles the input feature map into the robust feature with activations that help the model make correct predictions and the non-robust feature with activations that are responsible for model mispredictions upon adversarial attack. The Recalibration part then adjusts the non-robust activations to restore the potentially useful cues for model predictions. Extensive experiments verify the superiority of FSR compared to traditional deactivation techniques and demonstrate that it improves the robustness of existing adversarial training methods by up to 8.57% with small computational overhead. Codes are available at https://github.com/wkim97/FSR.



## **42. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2008.09312v3) [paper-pdf](http://arxiv.org/pdf/2008.09312v3)

**Authors**: Shiliang Zuo

**Abstract**: We consider a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. We propose a novel attack strategy that manipulates a UCB principle into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\sqrt{\log T}$, where $T$ is the number of rounds. We also prove the first lower bound on the cumulative attack cost. Our lower bound matches our upper bound up to $\log \log T$ factors, showing our attack to be near optimal.



## **43. RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit**

cs.LG

Published in Network and Distributed System Security (NDSS) Symposium  2022. Code is available at https://ramboattack.github.io/

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2112.05282v3) [paper-pdf](http://arxiv.org/pdf/2112.05282v3)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Machine learning models are critically susceptible to evasion attacks from adversarial examples. Generally, adversarial examples, modified inputs deceptively similar to the original input, are constructed under whitebox settings by adversaries with full access to the model. However, recent attacks have shown a remarkable reduction in query numbers to craft adversarial examples using blackbox attacks. Particularly, alarming is the ability to exploit the classification decision from the access interface of a trained model provided by a growing number of Machine Learning as a Service providers including Google, Microsoft, IBM and used by a plethora of applications incorporating these models. The ability of an adversary to exploit only the predicted label from a model to craft adversarial examples is distinguished as a decision-based attack. In our study, we first deep dive into recent state-of-the-art decision-based attacks in ICLR and SP to highlight the costly nature of discovering low distortion adversarial employing gradient estimation methods. We develop a robust query efficient attack capable of avoiding entrapment in a local minimum and misdirection from noisy gradients seen in gradient estimation methods. The attack method we propose, RamBoAttack, exploits the notion of Randomized Block Coordinate Descent to explore the hidden classifier manifold, targeting perturbations to manipulate only localized input features to address the issues of gradient estimation methods. Importantly, the RamBoAttack is more robust to the different sample inputs available to an adversary and the targeted class. Overall, for a given target class, RamBoAttack is demonstrated to be more robust at achieving a lower distortion within a given query budget. We curate our extensive results using the large-scale high-resolution ImageNet dataset and open-source our attack, test samples and artifacts on GitHub.



## **44. Query Efficient Decision Based Sparse Attacks Against Black-Box Deep Learning Models**

cs.LG

Published as a conference paper at the International Conference on  Learning Representations (ICLR 2022). Code is available at  https://sparseevoattack.github.io/

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2202.00091v2) [paper-pdf](http://arxiv.org/pdf/2202.00091v2)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Despite our best efforts, deep learning models remain highly vulnerable to even tiny adversarial perturbations applied to the inputs. The ability to extract information from solely the output of a machine learning model to craft adversarial perturbations to black-box models is a practical threat against real-world systems, such as autonomous cars or machine learning models exposed as a service (MLaaS). Of particular interest are sparse attacks. The realization of sparse attacks in black-box models demonstrates that machine learning models are more vulnerable than we believe. Because these attacks aim to minimize the number of perturbed pixels measured by l_0 norm-required to mislead a model by solely observing the decision (the predicted label) returned to a model query; the so-called decision-based attack setting. But, such an attack leads to an NP-hard optimization problem. We develop an evolution-based algorithm-SparseEvo-for the problem and evaluate against both convolutional deep neural networks and vision transformers. Notably, vision transformers are yet to be investigated under a decision-based attack setting. SparseEvo requires significantly fewer model queries than the state-of-the-art sparse attack Pointwise for both untargeted and targeted attacks. The attack algorithm, although conceptually simple, is also competitive with only a limited query budget against the state-of-the-art gradient-based whitebox attacks in standard computer vision tasks such as ImageNet. Importantly, the query efficient SparseEvo, along with decision-based attacks, in general, raise new questions regarding the safety of deployed systems and poses new directions to study and understand the robustness of machine learning models.



## **45. CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2302.13519v3) [paper-pdf](http://arxiv.org/pdf/2302.13519v3)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Patch-based physical attacks have increasingly aroused concerns.   However, most existing methods focus on obscuring targets captured on the ground, and some of these methods are simply extended to deceive aerial detectors.   They smear the targeted objects in the physical world with the elaborated adversarial patches, which can only slightly sway the aerial detectors' prediction and with weak attack transferability.   To address the above issues, we propose to perform Contextual Background Attack (CBA), a novel physical attack framework against aerial detection, which can achieve strong attack efficacy and transferability in the physical world even without smudging the interested objects at all.   Specifically, the targets of interest, i.e. the aircraft in aerial images, are adopted to mask adversarial patches.   The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world.   To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy.   Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously.   Extensive proportionally scaled experiments are performed in physical scenarios, demonstrating the superiority and potential of the proposed framework for physical attacks.   We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods.



## **46. TrojViT: Trojan Insertion in Vision Transformers**

cs.LG

10 pages, 4 figures, 11 tables

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2208.13049v3) [paper-pdf](http://arxiv.org/pdf/2208.13049v3)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstract**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.



## **47. Adversarial Robustness and Feature Impact Analysis for Driver Drowsiness Detection**

cs.LG

10 pages, 2 tables, 3 figures, AIME 2023 conference

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13649v1) [paper-pdf](http://arxiv.org/pdf/2303.13649v1)

**Authors**: João Vitorino, Lourenço Rodrigues, Eva Maia, Isabel Praça, André Lourenço

**Abstract**: Drowsy driving is a major cause of road accidents, but drivers are dismissive of the impact that fatigue can have on their reaction times. To detect drowsiness before any impairment occurs, a promising strategy is using Machine Learning (ML) to monitor Heart Rate Variability (HRV) signals. This work presents multiple experiments with different HRV time windows and ML models, a feature impact analysis using Shapley Additive Explanations (SHAP), and an adversarial robustness analysis to assess their reliability when processing faulty input data and perturbed HRV signals. The most reliable model was Extreme Gradient Boosting (XGB) and the optimal time window had between 120 and 150 seconds. Furthermore, SHAP enabled the selection of the 18 most impactful features and the training of new smaller models that achieved a performance as good as the initial ones. Despite the susceptibility of all models to adversarial attacks, adversarial training enabled them to preserve significantly higher results, especially XGB. Therefore, ML models can significantly benefit from realistic adversarial training to provide a more robust driver drowsiness detection.



## **48. Efficient Symbolic Reasoning for Neural-Network Verification**

cs.AI

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13588v1) [paper-pdf](http://arxiv.org/pdf/2303.13588v1)

**Authors**: Zi Wang, Somesh Jha, Krishnamurthy, Dvijotham

**Abstract**: The neural network has become an integral part of modern software systems. However, they still suffer from various problems, in particular, vulnerability to adversarial attacks. In this work, we present a novel program reasoning framework for neural-network verification, which we refer to as symbolic reasoning. The key components of our framework are the use of the symbolic domain and the quadratic relation. The symbolic domain has very flexible semantics, and the quadratic relation is quite expressive. They allow us to encode many verification problems for neural networks as quadratic programs. Our scheme then relaxes the quadratic programs to semidefinite programs, which can be efficiently solved. This framework allows us to verify various neural-network properties under different scenarios, especially those that appear challenging for non-symbolic domains. Moreover, it introduces new representations and perspectives for the verification tasks. We believe that our framework can bring new theoretical insights and practical tools to verification problems for neural networks.



## **49. Symmetries, flat minima, and the conserved quantities of gradient flow**

cs.LG

To appear at ICLR 2023

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2210.17216v2) [paper-pdf](http://arxiv.org/pdf/2210.17216v2)

**Authors**: Bo Zhao, Iordan Ganev, Robin Walters, Rose Yu, Nima Dehmamy

**Abstract**: Empirical studies of the loss landscape of deep networks have revealed that many local minima are connected through low-loss valleys. Yet, little is known about the theoretical origin of such valleys. We present a general framework for finding continuous symmetries in the parameter space, which carve out low-loss valleys. Our framework uses equivariances of the activation functions and can be applied to different layer architectures. To generalize this framework to nonlinear neural networks, we introduce a novel set of nonlinear, data-dependent symmetries. These symmetries can transform a trained model such that it performs similarly on new samples, which allows ensemble building that improves robustness under certain adversarial attacks. We then show that conserved quantities associated with linear symmetries can be used to define coordinates along low-loss valleys. The conserved quantities help reveal that using common initialization methods, gradient flow only explores a small part of the global minimum. By relating conserved quantities to convergence rate and sharpness of the minimum, we provide insights on how initialization impacts convergence and generalizability.



## **50. Decentralized Adversarial Training over Graphs**

cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13326v1) [paper-pdf](http://arxiv.org/pdf/2303.13326v1)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of diffusion learning, we develop a decentralized adversarial training framework for multi-agent systems. We analyze the convergence properties of the proposed scheme for both convex and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.



