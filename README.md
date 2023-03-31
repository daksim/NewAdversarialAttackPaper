# Latest Adversarial Attack Papers
**update at 2023-03-31 10:12:37**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. TorKameleon: Improving Tor's Censorship Resistance With K-anonimization and Media-based Covert Channels**

cs.CR

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.17544v1) [paper-pdf](http://arxiv.org/pdf/2303.17544v1)

**Authors**: João Afonso Vilalonga, João S. Resende, Henrique Domingos

**Abstract**: The use of anonymity networks such as Tor and similar tools can greatly enhance the privacy and anonymity of online communications. Tor, in particular, is currently the most widely used system for ensuring anonymity on the Internet. However, recent research has shown that Tor is vulnerable to correlation attacks carried out by state-level adversaries or colluding Internet censors. Therefore, new and more effective solutions emerged to protect online anonymity. Promising results have been achieved by implementing covert channels based on media traffic in modern anonymization systems, which have proven to be a reliable and practical approach to defend against powerful traffic correlation attacks. In this paper, we present TorKameleon, a censorship evasion solution that better protects Tor users from powerful traffic correlation attacks carried out by state-level adversaries. TorKameleon can be used either as a fully integrated Tor pluggable transport or as a standalone anonymization system that uses K-anonymization and encapsulation of user traffic in covert media channels. Our main goal is to protect users from machine and deep learning correlation attacks on anonymization networks like Tor. We have developed the TorKameleon prototype and performed extensive validations to verify the accuracy and experimental performance of the proposed solution in the Tor environment, including state-of-the-art active correlation attacks. As far as we know, we are the first to develop and study a system that uses both anonymization mechanisms described above against active correlation attacks.



## **2. Boosting Physical Layer Black-Box Attacks with Semantic Adversaries in Semantic Communications**

eess.SP

accepted by ICC2023

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.16523v2) [paper-pdf](http://arxiv.org/pdf/2303.16523v2)

**Authors**: Zeju Li, Xinghan Liu, Guoshun Nan, Jinfei Zhou, Xinchen Lyu, Qimei Cui, Xiaofeng Tao

**Abstract**: End-to-end semantic communication (ESC) system is able to improve communication efficiency by only transmitting the semantics of the input rather than raw bits. Although promising, ESC has also been shown susceptible to the crafted physical layer adversarial perturbations due to the openness of wireless channels and the sensitivity of neural models. Previous works focus more on the physical layer white-box attacks, while the challenging black-box ones, as more practical adversaries in real-world cases, are still largely under-explored. To this end, we present SemBLK, a novel method that can learn to generate destructive physical layer semantic attacks for an ESC system under the black-box setting, where the adversaries are imperceptible to humans. Specifically, 1) we first introduce a surrogate semantic encoder and train its parameters by exploring a limited number of queries to an existing ESC system. 2) Equipped with such a surrogate encoder, we then propose a novel semantic perturbation generation method to learn to boost the physical layer attacks with semantic adversaries. Experiments on two public datasets show the effectiveness of our proposed SemBLK in attacking the ESC system under the black-box setting. Finally, we provide case studies to visually justify the superiority of our physical layer semantic perturbations.



## **3. Non-Asymptotic Lower Bounds For Training Data Reconstruction**

cs.LG

Corrected minor typos and restructured appendix

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.16372v2) [paper-pdf](http://arxiv.org/pdf/2303.16372v2)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We investigate semantic guarantees of private learning algorithms for their resilience to training Data Reconstruction Attacks (DRAs) by informed adversaries. To this end, we derive non-asymptotic minimax lower bounds on the adversary's reconstruction error against learners that satisfy differential privacy (DP) and metric differential privacy (mDP). Furthermore, we demonstrate that our lower bound analysis for the latter also covers the high dimensional regime, wherein, the input data dimensionality may be larger than the adversary's query budget. Motivated by the theoretical improvements conferred by metric DP, we extend the privacy analysis of popular deep learning algorithms such as DP-SGD and Projected Noisy SGD to cover the broader notion of metric differential privacy.



## **4. Understanding the Robustness of 3D Object Detection with Bird's-Eye-View Representations in Autonomous Driving**

cs.CV

8 pages, CVPR2023

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.17297v1) [paper-pdf](http://arxiv.org/pdf/2303.17297v1)

**Authors**: Zijian Zhu, Yichi Zhang, Hai Chen, Yinpeng Dong, Shu Zhao, Wenbo Ding, Jiachen Zhong, Shibao Zheng

**Abstract**: 3D object detection is an essential perception task in autonomous driving to understand the environments. The Bird's-Eye-View (BEV) representations have significantly improved the performance of 3D detectors with camera inputs on popular benchmarks. However, there still lacks a systematic understanding of the robustness of these vision-dependent BEV models, which is closely related to the safety of autonomous driving systems. In this paper, we evaluate the natural and adversarial robustness of various representative models under extensive settings, to fully understand their behaviors influenced by explicit BEV features compared with those without BEV. In addition to the classic settings, we propose a 3D consistent patch attack by applying adversarial patches in the 3D space to guarantee the spatiotemporal consistency, which is more realistic for the scenario of autonomous driving. With substantial experiments, we draw several findings: 1) BEV models tend to be more stable than previous methods under different natural conditions and common corruptions due to the expressive spatial representations; 2) BEV models are more vulnerable to adversarial noises, mainly caused by the redundant BEV features; 3) Camera-LiDAR fusion models have superior performance under different settings with multi-modal inputs, but BEV fusion model is still vulnerable to adversarial noises of both point cloud and image. These findings alert the safety issue in the applications of BEV detectors and could facilitate the development of more robust models.



## **5. Adversarial Attack and Defense for Dehazing Networks**

cs.CV

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.17255v1) [paper-pdf](http://arxiv.org/pdf/2303.17255v1)

**Authors**: Jie Gui, Xiaofeng Cong, Chengwei Peng, Yuan Yan Tang, James Tin-Yau Kwok

**Abstract**: The research on single image dehazing task has been widely explored. However, as far as we know, no comprehensive study has been conducted on the robustness of the well-trained dehazing models. Therefore, there is no evidence that the dehazing networks can resist malicious attacks. In this paper, we focus on designing a group of attack methods based on first order gradient to verify the robustness of the existing dehazing algorithms. By analyzing the general goal of image dehazing task, five attack methods are proposed, which are prediction, noise, mask, ground-truth and input attack. The corresponding experiments are conducted on six datasets with different scales. Further, the defense strategy based on adversarial training is adopted for reducing the negative effects caused by malicious attacks. In summary, this paper defines a new challenging problem for image dehazing area, which can be called as adversarial attack on dehazing networks (AADN). Code is available at https://github.com/guijiejie/AADN.



## **6. A Tensor-based Convolutional Neural Network for Small Dataset Classification**

cs.CV

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.17061v1) [paper-pdf](http://arxiv.org/pdf/2303.17061v1)

**Authors**: Zhenhua Chen, David Crandall

**Abstract**: Inspired by the ConvNets with structured hidden representations, we propose a Tensor-based Neural Network, TCNN. Different from ConvNets, TCNNs are composed of structured neurons rather than scalar neurons, and the basic operation is neuron tensor transformation. Unlike other structured ConvNets, where the part-whole relationships are modeled explicitly, the relationships are learned implicitly in TCNNs. Also, the structured neurons in TCNNs are high-rank tensors rather than vectors or matrices. We compare TCNNs with current popular ConvNets, including ResNets, MobileNets, EfficientNets, RegNets, etc., on CIFAR10, CIFAR100, and Tiny ImageNet. The experiment shows that TCNNs have higher efficiency in terms of parameters. TCNNs also show higher robustness against white-box adversarial attacks on MNIST compared to ConvNets.



## **7. Beyond Empirical Risk Minimization: Local Structure Preserving Regularization for Improving Adversarial Robustness**

cs.LG

13 pages, 4 figures

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.16861v1) [paper-pdf](http://arxiv.org/pdf/2303.16861v1)

**Authors**: Wei Wei, Jiahuan Zhou, Ying Wu

**Abstract**: It is broadly known that deep neural networks are susceptible to being fooled by adversarial examples with perturbations imperceptible by humans. Various defenses have been proposed to improve adversarial robustness, among which adversarial training methods are most effective. However, most of these methods treat the training samples independently and demand a tremendous amount of samples to train a robust network, while ignoring the latent structural information among these samples. In this work, we propose a novel Local Structure Preserving (LSP) regularization, which aims to preserve the local structure of the input space in the learned embedding space. In this manner, the attacking effect of adversarial samples lying in the vicinity of clean samples can be alleviated. We show strong empirical evidence that with or without adversarial training, our method consistently improves the performance of adversarial robustness on several image classification datasets compared to the baselines and some state-of-the-art approaches, thus providing promising direction for future research.



## **8. Improving the Transferability of Adversarial Attacks on Face Recognition with Beneficial Perturbation Feature Augmentation**

cs.CV

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2210.16117v3) [paper-pdf](http://arxiv.org/pdf/2210.16117v3)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Zongyi Li, Ping Li

**Abstract**: Face recognition (FR) models can be easily fooled by adversarial examples, which are crafted by adding imperceptible perturbations on benign face images. The existence of adversarial face examples poses a great threat to the security of society. In order to build a more sustainable digital nation, in this paper, we improve the transferability of adversarial face examples to expose more blind spots of existing FR models. Though generating hard samples has shown its effectiveness in improving the generalization of models in training tasks, the effectiveness of utilizing this idea to improve the transferability of adversarial face examples remains unexplored. To this end, based on the property of hard samples and the symmetry between training tasks and adversarial attack tasks, we propose the concept of hard models, which have similar effects as hard samples for adversarial attack tasks. Utilizing the concept of hard models, we propose a novel attack method called Beneficial Perturbation Feature Augmentation Attack (BPFA), which reduces the overfitting of adversarial examples to surrogate FR models by constantly generating new hard models to craft the adversarial examples. Specifically, in the backpropagation, BPFA records the gradients on pre-selected feature maps and uses the gradient on the input image to craft the adversarial example. In the next forward propagation, BPFA leverages the recorded gradients to add beneficial perturbations on their corresponding feature maps to increase the loss. Extensive experiments demonstrate that BPFA can significantly boost the transferability of adversarial attacks on FR.



## **9. Imbalanced Gradients: A Subtle Cause of Overestimated Adversarial Robustness**

cs.CV

To appear in Machine Learning

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2006.13726v4) [paper-pdf](http://arxiv.org/pdf/2006.13726v4)

**Authors**: Xingjun Ma, Linxi Jiang, Hanxun Huang, Zejia Weng, James Bailey, Yu-Gang Jiang

**Abstract**: Evaluating the robustness of a defense model is a challenging task in adversarial robustness research. Obfuscated gradients have previously been found to exist in many defense methods and cause a false signal of robustness. In this paper, we identify a more subtle situation called Imbalanced Gradients that can also cause overestimated adversarial robustness. The phenomenon of imbalanced gradients occurs when the gradient of one term of the margin loss dominates and pushes the attack towards to a suboptimal direction. To exploit imbalanced gradients, we formulate a Margin Decomposition (MD) attack that decomposes a margin loss into individual terms and then explores the attackability of these terms separately via a two-stage process. We also propose a multi-targeted and ensemble version of our MD attack. By investigating 24 defense models proposed since 2018, we find that 11 models are susceptible to a certain degree of imbalanced gradients and our MD attack can decrease their robustness evaluated by the best standalone baseline attack by more than 1%. We also provide an in-depth investigation on the likely causes of imbalanced gradients and effective countermeasures. Our code is available at https://github.com/HanxunH/MDAttack.



## **10. Targeted Adversarial Attacks on Wind Power Forecasts**

cs.LG

20 pages, including appendix, 12 figures

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.16633v1) [paper-pdf](http://arxiv.org/pdf/2303.16633v1)

**Authors**: René Heinrich, Christoph Scholz, Stephan Vogt, Malte Lehna

**Abstract**: In recent years, researchers proposed a variety of deep learning models for wind power forecasting. These models predict the wind power generation of wind farms or entire regions more accurately than traditional machine learning algorithms or physical models. However, latest research has shown that deep learning models can often be manipulated by adversarial attacks. Since wind power forecasts are essential for the stability of modern power systems, it is important to protect them from this threat. In this work, we investigate the vulnerability of two different forecasting models to targeted, semitargeted, and untargeted adversarial attacks. We consider a Long Short-Term Memory (LSTM) network for predicting the power generation of a wind farm and a Convolutional Neural Network (CNN) for forecasting the wind power generation throughout Germany. Moreover, we propose the Total Adversarial Robustness Score (TARS), an evaluation metric for quantifying the robustness of regression models to targeted and semi-targeted adversarial attacks. It assesses the impact of attacks on the model's performance, as well as the extent to which the attacker's goal was achieved, by assigning a score between 0 (very vulnerable) and 1 (very robust). In our experiments, the LSTM forecasting model was fairly robust and achieved a TARS value of over 0.81 for all adversarial attacks investigated. The CNN forecasting model only achieved TARS values below 0.06 when trained ordinarily, and was thus very vulnerable. Yet, its robustness could be significantly improved by adversarial training, which always resulted in a TARS above 0.46.



## **11. Diffusion Denoised Smoothing for Certified and Adversarial Robust Out-Of-Distribution Detection**

cs.LG

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.14961v2) [paper-pdf](http://arxiv.org/pdf/2303.14961v2)

**Authors**: Nicola Franco, Daniel Korth, Jeanette Miriam Lorenz, Karsten Roscher, Stephan Guennemann

**Abstract**: As the use of machine learning continues to expand, the importance of ensuring its safety cannot be overstated. A key concern in this regard is the ability to identify whether a given sample is from the training distribution, or is an "Out-Of-Distribution" (OOD) sample. In addition, adversaries can manipulate OOD samples in ways that lead a classifier to make a confident prediction. In this study, we present a novel approach for certifying the robustness of OOD detection within a $\ell_2$-norm around the input, regardless of network architecture and without the need for specific components or additional training. Further, we improve current techniques for detecting adversarial attacks on OOD samples, while providing high levels of certified and adversarial robustness on in-distribution samples. The average of all OOD detection metrics on CIFAR10/100 shows an increase of $\sim 13 \% / 5\%$ relative to previous approaches.



## **12. What Does the Gradient Tell When Attacking the Graph Structure**

cs.LG

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2208.12815v2) [paper-pdf](http://arxiv.org/pdf/2208.12815v2)

**Authors**: Zihan Liu, Ge Wang, Yun Luo, Stan Z. Li

**Abstract**: Recent research has revealed that Graph Neural Networks (GNNs) are susceptible to adversarial attacks targeting the graph structure. A malicious attacker can manipulate a limited number of edges, given the training labels, to impair the victim model's performance. Previous empirical studies indicate that gradient-based attackers tend to add edges rather than remove them. In this paper, we present a theoretical demonstration revealing that attackers tend to increase inter-class edges due to the message passing mechanism of GNNs, which explains some previous empirical observations. By connecting dissimilar nodes, attackers can more effectively corrupt node features, making such attacks more advantageous. However, we demonstrate that the inherent smoothness of GNN's message passing tends to blur node dissimilarity in the feature space, leading to the loss of crucial information during the forward process. To address this issue, we propose a novel surrogate model with multi-level propagation that preserves the node dissimilarity information. This model parallelizes the propagation of unaggregated raw features and multi-hop aggregated features, while introducing batch normalization to enhance the dissimilarity in node representations and counteract the smoothness resulting from topological aggregation. Our experiments show significant improvement with our approach.Furthermore, both theoretical and experimental evidence suggest that adding inter-class edges constitutes an easily observable attack pattern. We propose an innovative attack loss that balances attack effectiveness and imperceptibility, sacrificing some attack effectiveness to attain greater imperceptibility. We also provide experiments to validate the compromise performance achieved through this attack loss.



## **13. A Pilot Study of Query-Free Adversarial Attack against Stable Diffusion**

cs.CV

The 3rd Workshop of Adversarial Machine Learning on Computer Vision:  Art of Robustness

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.16378v1) [paper-pdf](http://arxiv.org/pdf/2303.16378v1)

**Authors**: Haomin Zhuang, Yihua Zhang, Sijia Liu

**Abstract**: Despite the record-breaking performance in Text-to-Image (T2I) generation by Stable Diffusion, less research attention is paid to its adversarial robustness. In this work, we study the problem of adversarial attack generation for Stable Diffusion and ask if an adversarial text prompt can be obtained even in the absence of end-to-end model queries. We call the resulting problem 'query-free attack generation'. To resolve this problem, we show that the vulnerability of T2I models is rooted in the lack of robustness of text encoders, e.g., the CLIP text encoder used for attacking Stable Diffusion. Based on such insight, we propose both untargeted and targeted query-free attacks, where the former is built on the most influential dimensions in the text embedding space, which we call steerable key dimensions. By leveraging the proposed attacks, we empirically show that only a five-character perturbation to the text prompt is able to cause the significant content shift of synthesized images using Stable Diffusion. Moreover, we show that the proposed target attack can precisely steer the diffusion model to scrub the targeted image content without causing much change in untargeted image content.



## **14. A Survey on Malware Detection with Graph Representation Learning**

cs.CR

Preprint, submitted to ACM Computing Surveys on March 2023. For any  suggestions or improvements, please contact me directly by e-mail

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.16004v1) [paper-pdf](http://arxiv.org/pdf/2303.16004v1)

**Authors**: Tristan Bilot, Nour El Madhoun, Khaldoun Al Agha, Anis Zouaoui

**Abstract**: Malware detection has become a major concern due to the increasing number and complexity of malware. Traditional detection methods based on signatures and heuristics are used for malware detection, but unfortunately, they suffer from poor generalization to unknown attacks and can be easily circumvented using obfuscation techniques. In recent years, Machine Learning (ML) and notably Deep Learning (DL) achieved impressive results in malware detection by learning useful representations from data and have become a solution preferred over traditional methods. More recently, the application of such techniques on graph-structured data has achieved state-of-the-art performance in various domains and demonstrates promising results in learning more robust representations from malware. Yet, no literature review focusing on graph-based deep learning for malware detection exists. In this survey, we provide an in-depth literature review to summarize and unify existing works under the common approaches and architectures. We notably demonstrate that Graph Neural Networks (GNNs) reach competitive results in learning robust embeddings from malware represented as expressive graph structures, leading to an efficient detection by downstream classifiers. This paper also reviews adversarial attacks that are utilized to fool graph-based detection methods. Challenges and future research directions are discussed at the end of the paper.



## **15. TransAudio: Towards the Transferable Adversarial Audio Attack via Learning Contextualized Perturbations**

cs.SD

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15940v1) [paper-pdf](http://arxiv.org/pdf/2303.15940v1)

**Authors**: Qi Gege, Yuefeng Chen, Xiaofeng Mao, Yao Zhu, Binyuan Hui, Xiaodan Li, Rong Zhang, Hui Xue

**Abstract**: In a transfer-based attack against Automatic Speech Recognition (ASR) systems, attacks are unable to access the architecture and parameters of the target model. Existing attack methods are mostly investigated in voice assistant scenarios with restricted voice commands, prohibiting their applicability to more general ASR related applications. To tackle this challenge, we propose a novel contextualized attack with deletion, insertion, and substitution adversarial behaviors, namely TransAudio, which achieves arbitrary word-level attacks based on the proposed two-stage framework. To strengthen the attack transferability, we further introduce an audio score-matching optimization strategy to regularize the training process, which mitigates adversarial example over-fitting to the surrogate model. Extensive experiments and analysis demonstrate the effectiveness of TransAudio against open-source ASR models and commercial APIs.



## **16. Denoising Autoencoder-based Defensive Distillation as an Adversarial Robustness Algorithm**

cs.LG

This paper have 4 pages, 3 figures and it is accepted at the Ada User  journal

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15901v1) [paper-pdf](http://arxiv.org/pdf/2303.15901v1)

**Authors**: Bakary Badjie, José Cecílio, António Casimiro

**Abstract**: Adversarial attacks significantly threaten the robustness of deep neural networks (DNNs). Despite the multiple defensive methods employed, they are nevertheless vulnerable to poison attacks, where attackers meddle with the initial training data. In order to defend DNNs against such adversarial attacks, this work proposes a novel method that combines the defensive distillation mechanism with a denoising autoencoder (DAE). This technique tries to lower the sensitivity of the distilled model to poison attacks by spotting and reconstructing poisonous adversarial inputs in the training data. We added carefully created adversarial samples to the initial training data to assess the proposed method's performance. Our experimental findings demonstrate that our method successfully identified and reconstructed the poisonous inputs while also considering enhancing the DNN's resilience. The proposed approach provides a potent and robust defense mechanism for DNNs in various applications where data poisoning attacks are a concern. Thus, the defensive distillation technique's limitation posed by poisonous adversarial attacks is overcome.



## **17. Towards Effective Adversarial Textured 3D Meshes on Physical Face Recognition**

cs.CV

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15818v1) [paper-pdf](http://arxiv.org/pdf/2303.15818v1)

**Authors**: Xiao Yang, Chang Liu, Longlong Xu, Yikai Wang, Yinpeng Dong, Ning Chen, Hang Su, Jun Zhu

**Abstract**: Face recognition is a prevailing authentication solution in numerous biometric applications. Physical adversarial attacks, as an important surrogate, can identify the weaknesses of face recognition systems and evaluate their robustness before deployed. However, most existing physical attacks are either detectable readily or ineffective against commercial recognition systems. The goal of this work is to develop a more reliable technique that can carry out an end-to-end evaluation of adversarial robustness for commercial systems. It requires that this technique can simultaneously deceive black-box recognition models and evade defensive mechanisms. To fulfill this, we design adversarial textured 3D meshes (AT3D) with an elaborate topology on a human face, which can be 3D-printed and pasted on the attacker's face to evade the defenses. However, the mesh-based optimization regime calculates gradients in high-dimensional mesh space, and can be trapped into local optima with unsatisfactory transferability. To deviate from the mesh-based space, we propose to perturb the low-dimensional coefficient space based on 3D Morphable Model, which significantly improves black-box transferability meanwhile enjoying faster search efficiency and better visual quality. Extensive experiments in digital and physical scenarios show that our method effectively explores the security vulnerabilities of multiple popular commercial services, including three recognition APIs, four anti-spoofing APIs, two prevailing mobile phones and two automated access control systems.



## **18. Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization**

cs.CV

CVPR 2023

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15754v1) [paper-pdf](http://arxiv.org/pdf/2303.15754v1)

**Authors**: Jianping Zhang, Yizhan Huang, Weibin Wu, Michael R. Lyu

**Abstract**: Vision transformers (ViTs) have been successfully deployed in a variety of computer vision tasks, but they are still vulnerable to adversarial samples. Transfer-based attacks use a local model to generate adversarial samples and directly transfer them to attack a target black-box model. The high efficiency of transfer-based attacks makes it a severe security threat to ViT-based applications. Therefore, it is vital to design effective transfer-based attacks to identify the deficiencies of ViTs beforehand in security-sensitive scenarios. Existing efforts generally focus on regularizing the input gradients to stabilize the updated direction of adversarial samples. However, the variance of the back-propagated gradients in intermediate blocks of ViTs may still be large, which may make the generated adversarial samples focus on some model-specific features and get stuck in poor local optima. To overcome the shortcomings of existing approaches, we propose the Token Gradient Regularization (TGR) method. According to the structural characteristics of ViTs, TGR reduces the variance of the back-propagated gradient in each internal block of ViTs in a token-wise manner and utilizes the regularized gradient to generate adversarial samples. Extensive experiments on attacking both ViTs and CNNs confirm the superiority of our approach. Notably, compared to the state-of-the-art transfer-based attacks, our TGR offers a performance improvement of 8.8% on average.



## **19. Improving the Transferability of Adversarial Samples by Path-Augmented Method**

cs.CV

10 pages + appendix, CVPR 2023

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15735v1) [paper-pdf](http://arxiv.org/pdf/2303.15735v1)

**Authors**: Jianping Zhang, Jen-tse Huang, Wenxuan Wang, Yichen Li, Weibin Wu, Xiaosen Wang, Yuxin Su, Michael R. Lyu

**Abstract**: Deep neural networks have achieved unprecedented success on diverse vision tasks. However, they are vulnerable to adversarial noise that is imperceptible to humans. This phenomenon negatively affects their deployment in real-world scenarios, especially security-related ones. To evaluate the robustness of a target model in practice, transfer-based attacks craft adversarial samples with a local model and have attracted increasing attention from researchers due to their high efficiency. The state-of-the-art transfer-based attacks are generally based on data augmentation, which typically augments multiple training images from a linear path when learning adversarial samples. However, such methods selected the image augmentation path heuristically and may augment images that are semantics-inconsistent with the target images, which harms the transferability of the generated adversarial samples. To overcome the pitfall, we propose the Path-Augmented Method (PAM). Specifically, PAM first constructs a candidate augmentation path pool. It then settles the employed augmentation paths during adversarial sample generation with greedy search. Furthermore, to avoid augmenting semantics-inconsistent images, we train a Semantics Predictor (SP) to constrain the length of the augmentation path. Extensive experiments confirm that PAM can achieve an improvement of over 4.8% on average compared with the state-of-the-art baselines in terms of the attack success rates.



## **20. EMShepherd: Detecting Adversarial Samples via Side-channel Leakage**

cs.CR

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15571v1) [paper-pdf](http://arxiv.org/pdf/2303.15571v1)

**Authors**: Ruyi Ding, Cheng Gongye, Siyue Wang, Aidong Ding, Yunsi Fei

**Abstract**: Deep Neural Networks (DNN) are vulnerable to adversarial perturbations-small changes crafted deliberately on the input to mislead the model for wrong predictions. Adversarial attacks have disastrous consequences for deep learning-empowered critical applications. Existing defense and detection techniques both require extensive knowledge of the model, testing inputs, and even execution details. They are not viable for general deep learning implementations where the model internal is unknown, a common 'black-box' scenario for model users. Inspired by the fact that electromagnetic (EM) emanations of a model inference are dependent on both operations and data and may contain footprints of different input classes, we propose a framework, EMShepherd, to capture EM traces of model execution, perform processing on traces and exploit them for adversarial detection. Only benign samples and their EM traces are used to train the adversarial detector: a set of EM classifiers and class-specific unsupervised anomaly detectors. When the victim model system is under attack by an adversarial example, the model execution will be different from executions for the known classes, and the EM trace will be different. We demonstrate that our air-gapped EMShepherd can effectively detect different adversarial attacks on a commonly used FPGA deep learning accelerator for both Fashion MNIST and CIFAR-10 datasets. It achieves a 100% detection rate on most types of adversarial samples, which is comparable to the state-of-the-art 'white-box' software-based detectors.



## **21. Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder**

cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15564v1) [paper-pdf](http://arxiv.org/pdf/2303.15564v1)

**Authors**: Tao Sun, Lu Pang, Chao Chen, Haibin Ling

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, where an adversary maliciously manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which are impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for black-box models. The true label of every test image needs to be recovered on the fly from the hard label predictions of a suspicious model. The heuristic trigger search in image space, however, is not scalable to complex triggers or high image resolution. We circumvent such barrier by leveraging generic image generation models, and propose a framework of Blind Defense with Masked AutoEncoder (BDMAE). It uses the image structural similarity and label consistency between the test image and MAE restorations to detect possible triggers. The detection result is refined by considering the topology of triggers. We obtain a purified test image from restorations for making prediction. Our approach is blind to the model architectures, trigger patterns or image benignity. Extensive experiments on multiple datasets with different backdoor attacks validate its effectiveness and generalizability. Code is available at https://github.com/tsun/BDMAE.



## **22. Intel TDX Demystified: A Top-Down Approach**

cs.CR

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15540v1) [paper-pdf](http://arxiv.org/pdf/2303.15540v1)

**Authors**: Pau-Chen Cheng, Wojciech Ozga, Enriquillo Valdez, Salman Ahmed, Zhongshu Gu, Hani Jamjoom, Hubertus Franke, James Bottomley

**Abstract**: Intel Trust Domain Extensions (TDX) is a new architectural extension in the 4th Generation Intel Xeon Scalable Processor that supports confidential computing. TDX allows the deployment of virtual machines in the Secure-Arbitration Mode (SEAM) with encrypted CPU state and memory, integrity protection, and remote attestation. TDX aims to enforce hardware-assisted isolation for virtual machines and minimize the attack surface exposed to host platforms, which are considered to be untrustworthy or adversarial in the confidential computing's new threat model. TDX can be leveraged by regulated industries or sensitive data holders to outsource their computations and data with end-to-end protection in public cloud infrastructure.   This paper aims to provide a comprehensive understanding of TDX to potential adopters, domain experts, and security researchers looking to leverage the technology for their own purposes. We adopt a top-down approach, starting with high-level security principles and moving to low-level technical details of TDX. Our analysis is based on publicly available documentation and source code, offering insights from security researchers outside of Intel.



## **23. Classifier Robustness Enhancement Via Test-Time Transformation**

cs.CV

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15409v1) [paper-pdf](http://arxiv.org/pdf/2303.15409v1)

**Authors**: Tsachi Blau, Roy Ganz, Chaim Baskin, Michael Elad, Alex Bronstein

**Abstract**: It has been recently discovered that adversarially trained classifiers exhibit an intriguing property, referred to as perceptually aligned gradients (PAG). PAG implies that the gradients of such classifiers possess a meaningful structure, aligned with human perception. Adversarial training is currently the best-known way to achieve classification robustness under adversarial attacks. The PAG property, however, has yet to be leveraged for further improving classifier robustness. In this work, we introduce Classifier Robustness Enhancement Via Test-Time Transformation (TETRA) -- a novel defense method that utilizes PAG, enhancing the performance of trained robust classifiers. Our method operates in two phases. First, it modifies the input image via a designated targeted adversarial attack into each of the dataset's classes. Then, it classifies the input image based on the distance to each of the modified instances, with the assumption that the shortest distance relates to the true class. We show that the proposed method achieves state-of-the-art results and validate our claim through extensive experiments on a variety of defense methods, classifier architectures, and datasets. We also empirically demonstrate that TETRA can boost the accuracy of any differentiable adversarial training classifier across a variety of attacks, including ones unseen at training. Specifically, applying TETRA leads to substantial improvement of up to $+23\%$, $+20\%$, and $+26\%$ on CIFAR10, CIFAR100, and ImageNet, respectively.



## **24. Learning the Unlearnable: Adversarial Augmentations Suppress Unlearnable Example Attacks**

cs.LG

UEraser introduces adversarial augmentations to suppress unlearnable  example attacks and outperforms current defenses

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15127v1) [paper-pdf](http://arxiv.org/pdf/2303.15127v1)

**Authors**: Tianrui Qin, Xitong Gao, Juanjuan Zhao, Kejiang Ye, Cheng-Zhong Xu

**Abstract**: Unlearnable example attacks are data poisoning techniques that can be used to safeguard public data against unauthorized use for training deep learning models. These methods add stealthy perturbations to the original image, thereby making it difficult for deep learning models to learn from these training data effectively. Current research suggests that adversarial training can, to a certain degree, mitigate the impact of unlearnable example attacks, while common data augmentation methods are not effective against such poisons. Adversarial training, however, demands considerable computational resources and can result in non-trivial accuracy loss. In this paper, we introduce the UEraser method, which outperforms current defenses against different types of state-of-the-art unlearnable example attacks through a combination of effective data augmentation policies and loss-maximizing adversarial augmentations. In stark contrast to the current SOTA adversarial training methods, UEraser uses adversarial augmentations, which extends beyond the confines of $ \ell_p $ perturbation budget assumed by current unlearning attacks and defenses. It also helps to improve the model's generalization ability, thus protecting against accuracy loss. UEraser wipes out the unlearning effect with error-maximizing data augmentations, thus restoring trained model accuracies. Interestingly, UEraser-Lite, a fast variant without adversarial augmentations, is also highly effective in preserving clean accuracies. On challenging unlearnable CIFAR-10, CIFAR-100, SVHN, and ImageNet-subset datasets produced with various attacks, it achieves results that are comparable to those obtained during clean training. We also demonstrate its efficacy against possible adaptive attacks. Our code is open source and available to the deep learning community: https://github.com/lafeat/ueraser.



## **25. Among Us: Adversarially Robust Collaborative Perception by Consensus**

cs.RO

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.09495v2) [paper-pdf](http://arxiv.org/pdf/2303.09495v2)

**Authors**: Yiming Li, Qi Fang, Jiamu Bai, Siheng Chen, Felix Juefei-Xu, Chen Feng

**Abstract**: Multiple robots could perceive a scene (e.g., detect objects) collaboratively better than individuals, although easily suffer from adversarial attacks when using deep learning. This could be addressed by the adversarial defense, but its training requires the often-unknown attacking mechanism. Differently, we propose ROBOSAC, a novel sampling-based defense strategy generalizable to unseen attackers. Our key idea is that collaborative perception should lead to consensus rather than dissensus in results compared to individual perception. This leads to our hypothesize-and-verify framework: perception results with and without collaboration from a random subset of teammates are compared until reaching a consensus. In such a framework, more teammates in the sampled subset often entail better perception performance but require longer sampling time to reject potential attackers. Thus, we derive how many sampling trials are needed to ensure the desired size of an attacker-free subset, or equivalently, the maximum size of such a subset that we can successfully sample within a given number of trials. We validate our method on the task of collaborative 3D object detection in autonomous driving scenarios.



## **26. Identifying Adversarially Attackable and Robust Samples**

cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2301.12896v2) [paper-pdf](http://arxiv.org/pdf/2301.12896v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Adversarial attacks insert small, imperceptible perturbations to input samples that cause large, undesired changes to the output of deep learning models. Despite extensive research on generating adversarial attacks and building defense systems, there has been limited research on understanding adversarial attacks from an input-data perspective. This work introduces the notion of sample attackability, where we aim to identify samples that are most susceptible to adversarial attacks (attackable samples) and conversely also identify the least susceptible samples (robust samples). We propose a deep-learning-based method to detect the adversarially attackable and robust samples in an unseen dataset for an unseen target model. Experiments on standard image classification datasets enables us to assess the portability of the deep attackability detector across a range of architectures. We find that the deep attackability detector performs better than simple model uncertainty-based measures for identifying the attackable/robust samples. This suggests that uncertainty is an inadequate proxy for measuring sample distance to a decision boundary. In addition to better understanding adversarial attack theory, it is found that the ability to identify the adversarially attackable and robust samples has implications for improving the efficiency of sample-selection tasks, e.g. active learning in augmentation for adversarial training.



## **27. Improving the Transferability of Adversarial Examples via Direction Tuning**

cs.CV

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15109v1) [paper-pdf](http://arxiv.org/pdf/2303.15109v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstract**: In the transfer-based adversarial attacks, adversarial examples are only generated by the surrogate models and achieve effective perturbation in the victim models. Although considerable efforts have been developed on improving the transferability of adversarial examples generated by transfer-based adversarial attacks, our investigation found that, the big deviation between the actual and steepest update directions of the current transfer-based adversarial attacks is caused by the large update step length, resulting in the generated adversarial examples can not converge well. However, directly reducing the update step length will lead to serious update oscillation so that the generated adversarial examples also can not achieve great transferability to the victim models. To address these issues, a novel transfer-based attack, namely direction tuning attack, is proposed to not only decrease the update deviation in the large step length, but also mitigate the update oscillation in the small sampling step length, thereby making the generated adversarial examples converge well to achieve great transferability on victim models. In addition, a network pruning method is proposed to smooth the decision boundary, thereby further decreasing the update oscillation and enhancing the transferability of the generated adversarial examples. The experiment results on ImageNet demonstrate that the average attack success rate (ASR) of the adversarial examples generated by our method can be improved from 87.9\% to 94.5\% on five victim models without defenses, and from 69.1\% to 76.2\% on eight advanced defense methods, in comparison with that of latest gradient-based attacks.



## **28. Improved Adversarial Training Through Adaptive Instance-wise Loss Smoothing**

cs.CV

12 pages, work in submission

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14077v2) [paper-pdf](http://arxiv.org/pdf/2303.14077v2)

**Authors**: Lin Li, Michael Spratling

**Abstract**: Deep neural networks can be easily fooled into making incorrect predictions through corruption of the input by adversarial perturbations: human-imperceptible artificial noise. So far adversarial training has been the most successful defense against such adversarial attacks. This work focuses on improving adversarial training to boost adversarial robustness. We first analyze, from an instance-wise perspective, how adversarial vulnerability evolves during adversarial training. We find that during training an overall reduction of adversarial loss is achieved by sacrificing a considerable proportion of training samples to be more vulnerable to adversarial attack, which results in an uneven distribution of adversarial vulnerability among data. Such "uneven vulnerability", is prevalent across several popular robust training methods and, more importantly, relates to overfitting in adversarial training. Motivated by this observation, we propose a new adversarial training method: Instance-adaptive Smoothness Enhanced Adversarial Training (ISEAT). It jointly smooths both input and weight loss landscapes in an adaptive, instance-specific, way to enhance robustness more for those samples with higher adversarial vulnerability. Extensive experiments demonstrate the superiority of our method over existing defense methods. Noticeably, our method, when combined with the latest data augmentation and semi-supervised learning techniques, achieves state-of-the-art robustness against $\ell_{\infty}$-norm constrained attacks on CIFAR10 of 59.32% for Wide ResNet34-10 without extra data, and 61.55% for Wide ResNet28-10 with extra data. Code is available at https://github.com/TreeLLi/Instance-adaptive-Smoothness-Enhanced-AT.



## **29. CAT:Collaborative Adversarial Training**

cs.CV

Tech report

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14922v1) [paper-pdf](http://arxiv.org/pdf/2303.14922v1)

**Authors**: Xingbin Liu, Huafeng Kuang, Xianming Lin, Yongjian Wu, Rongrong Ji

**Abstract**: Adversarial training can improve the robustness of neural networks. Previous methods focus on a single adversarial training strategy and do not consider the model property trained by different strategies. By revisiting the previous methods, we find different adversarial training methods have distinct robustness for sample instances. For example, a sample instance can be correctly classified by a model trained using standard adversarial training (AT) but not by a model trained using TRADES, and vice versa. Based on this observation, we propose a collaborative adversarial training framework to improve the robustness of neural networks. Specifically, we use different adversarial training methods to train robust models and let models interact with their knowledge during the training process. Collaborative Adversarial Training (CAT) can improve both robustness and accuracy. Extensive experiments on various networks and datasets validate the effectiveness of our method. CAT achieves state-of-the-art adversarial robustness without using any additional data on CIFAR-10 under the Auto-Attack benchmark. Code is available at https://github.com/liuxingbin/CAT.



## **30. Efficient Robustness Assessment via Adversarial Spatial-Temporal Focus on Videos**

cs.CV

accepted by TPAMI2023

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2301.00896v2) [paper-pdf](http://arxiv.org/pdf/2301.00896v2)

**Authors**: Wei Xingxing, Wang Songping, Yan Huanqian

**Abstract**: Adversarial robustness assessment for video recognition models has raised concerns owing to their wide applications on safety-critical tasks. Compared with images, videos have much high dimension, which brings huge computational costs when generating adversarial videos. This is especially serious for the query-based black-box attacks where gradient estimation for the threat models is usually utilized, and high dimensions will lead to a large number of queries. To mitigate this issue, we propose to simultaneously eliminate the temporal and spatial redundancy within the video to achieve an effective and efficient gradient estimation on the reduced searching space, and thus query number could decrease. To implement this idea, we design the novel Adversarial spatial-temporal Focus (AstFocus) attack on videos, which performs attacks on the simultaneously focused key frames and key regions from the inter-frames and intra-frames in the video. AstFocus attack is based on the cooperative Multi-Agent Reinforcement Learning (MARL) framework. One agent is responsible for selecting key frames, and another agent is responsible for selecting key regions. These two agents are jointly trained by the common rewards received from the black-box threat models to perform a cooperative prediction. By continuously querying, the reduced searching space composed of key frames and key regions is becoming precise, and the whole query number becomes less than that on the original video. Extensive experiments on four mainstream video recognition models and three widely used action recognition datasets demonstrate that the proposed AstFocus attack outperforms the SOTA methods, which is prevenient in fooling rate, query number, time, and perturbation magnitude at the same.



## **31. Don't be a Victim During a Pandemic! Analysing Security and Privacy Threats in Twitter During COVID-19**

cs.CR

Paper has been accepted for publication in IEEE Access. Currently  available on IEEE ACCESS early access (see DOI)

**SubmitDate**: 2023-03-26    [abs](http://arxiv.org/abs/2202.10543v2) [paper-pdf](http://arxiv.org/pdf/2202.10543v2)

**Authors**: Bibhas Sharma, Ishan Karunanayake, Rahat Masood, Muhammad Ikram

**Abstract**: There has been a huge spike in the usage of social media platforms during the COVID-19 lockdowns. These lockdown periods have resulted in a set of new cybercrimes, thereby allowing attackers to victimise social media users with a range of threats. This paper performs a large-scale study to investigate the impact of a pandemic and the lockdown periods on the security and privacy of social media users. We analyse 10.6 Million COVID-related tweets from 533 days of data crawling and investigate users' security and privacy behaviour in three different periods (i.e., before, during, and after the lockdown). Our study shows that users unintentionally share more personal identifiable information when writing about the pandemic situation (e.g., sharing nearby coronavirus testing locations) in their tweets. The privacy risk reaches 100% if a user posts three or more sensitive tweets about the pandemic. We investigate the number of suspicious domains shared on social media during different phases of the pandemic. Our analysis reveals an increase in the number of suspicious domains during the lockdown compared to other lockdown phases. We observe that IT, Search Engines, and Businesses are the top three categories that contain suspicious domains. Our analysis reveals that adversaries' strategies to instigate malicious activities change with the country's pandemic situation.



## **32. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

stat.ML

9 pages; Previously this version appeared as arXiv:2210.08198 which  was submitted as a new work by accident

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2109.12772v2) [paper-pdf](http://arxiv.org/pdf/2109.12772v2)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.



## **33. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

cs.CV

This work was intended as a replacement of arXiv:2109.12772 and any  subsequent updates will appear there

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2210.08198v2) [paper-pdf](http://arxiv.org/pdf/2210.08198v2)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Ch. Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.



## **34. STDLens: Model Hijacking-Resilient Federated Learning for Object Detection**

cs.CR

CVPR 2023. Source Code: https://github.com/git-disl/STDLens

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.11511v2) [paper-pdf](http://arxiv.org/pdf/2303.11511v2)

**Authors**: Ka-Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu

**Abstract**: Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates.



## **35. Improving robustness of jet tagging algorithms with adversarial training: exploring the loss surface**

hep-ex

5 pages, 2 figures; submitted to ACAT 2022 proceedings

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14511v1) [paper-pdf](http://arxiv.org/pdf/2303.14511v1)

**Authors**: Annika Stein

**Abstract**: In the field of high-energy physics, deep learning algorithms continue to gain in relevance and provide performance improvements over traditional methods, for example when identifying rare signals or finding complex patterns. From an analyst's perspective, obtaining highest possible performance is desirable, but recently, some attention has been shifted towards studying robustness of models to investigate how well these perform under slight distortions of input features. Especially for tasks that involve many (low-level) inputs, the application of deep neural networks brings new challenges. In the context of jet flavor tagging, adversarial attacks are used to probe a typical classifier's vulnerability and can be understood as a model for systematic uncertainties. A corresponding defense strategy, adversarial training, improves robustness, while maintaining high performance. Investigating the loss surface corresponding to the inputs and models in question reveals geometric interpretations of robustness, taking correlations into account.



## **36. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

cs.CV

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2204.10779v6) [paper-pdf](http://arxiv.org/pdf/2204.10779v6)

**Authors**: Xunguang Wang, Yiqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61\%, 12.35\%, and 11.56\% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively. The code is available at https://github.com/xunguangwang/CgAT.



## **37. No more Reviewer #2: Subverting Automatic Paper-Reviewer Assignment using Adversarial Learning**

cs.CR

Accepted at USENIX Security Symposium 2023

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14443v1) [paper-pdf](http://arxiv.org/pdf/2303.14443v1)

**Authors**: Thorsten Eisenhofer, Erwin Quiring, Jonas Möller, Doreen Riepel, Thorsten Holz, Konrad Rieck

**Abstract**: The number of papers submitted to academic conferences is steadily rising in many scientific disciplines. To handle this growth, systems for automatic paper-reviewer assignments are increasingly used during the reviewing process. These systems use statistical topic models to characterize the content of submissions and automate the assignment to reviewers. In this paper, we show that this automation can be manipulated using adversarial learning. We propose an attack that adapts a given paper so that it misleads the assignment and selects its own reviewers. Our attack is based on a novel optimization strategy that alternates between the feature space and problem space to realize unobtrusive changes to the paper. To evaluate the feasibility of our attack, we simulate the paper-reviewer assignment of an actual security conference (IEEE S&P) with 165 reviewers on the program committee. Our results show that we can successfully select and remove reviewers without access to the assignment system. Moreover, we demonstrate that the manipulated papers remain plausible and are often indistinguishable from benign submissions.



## **38. A User-Based Authentication and DoS Mitigation Scheme for Wearable Wireless Body Sensor Networks**

cs.CR

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14441v1) [paper-pdf](http://arxiv.org/pdf/2303.14441v1)

**Authors**: Nombulelo Zulu, Deon P. Du Plessis, Topside E. Mathonsi, Tshimangadzo M. Tshilongamulenzhe

**Abstract**: Wireless Body Sensor Networks (WBSNs) is one of the greatest growing technology for sensing and performing various tasks. The information transmitted in the WBSNs is vulnerable to cyber-attacks, therefore security is very important. Denial of Service (DoS) attacks are considered one of the major threats against WBSNs security. In DoS attacks, an adversary targets to degrade and shut down the efficient use of the network and disrupt the services in the network causing them inaccessible to its intended users. If sensitive information of patients in WBSNs, such as the medical history is accessed by unauthorized users, the patient may suffer much more than the disease itself, it may result in loss of life. This paper proposes a User-Based authentication scheme to mitigate DoS attacks in WBSNs. A five-phase User-Based authentication DoS mitigation scheme for WBSNs is designed by integrating Elliptic Curve Cryptography (ECC) with Rivest Cipher 4 (RC4) to ensure a strong authentication process that will only allow authorized users to access nodes on WBSNs.



## **39. Consistent Attack: Universal Adversarial Perturbation on Embodied Vision Navigation**

cs.LG

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2206.05751v4) [paper-pdf](http://arxiv.org/pdf/2206.05751v4)

**Authors**: Chengyang Ying, You Qiaoben, Xinning Zhou, Hang Su, Wenbo Ding, Jianyong Ai

**Abstract**: Embodied agents in vision navigation coupled with deep neural networks have attracted increasing attention. However, deep neural networks have been shown vulnerable to malicious adversarial noises, which may potentially cause catastrophic failures in Embodied Vision Navigation. Among different adversarial noises, universal adversarial perturbations (UAP), i.e., a constant image-agnostic perturbation applied on every input frame of the agent, play a critical role in Embodied Vision Navigation since they are computation-efficient and application-practical during the attack. However, existing UAP methods ignore the system dynamics of Embodied Vision Navigation and might be sub-optimal. In order to extend UAP to the sequential decision setting, we formulate the disturbed environment under the universal noise $\delta$, as a $\delta$-disturbed Markov Decision Process ($\delta$-MDP). Based on the formulation, we analyze the properties of $\delta$-MDP and propose two novel Consistent Attack methods, named Reward UAP and Trajectory UAP, for attacking Embodied agents, which consider the dynamic of the MDP and calculate universal noises by estimating the disturbed distribution and the disturbed Q function. For various victim models, our Consistent Attack can cause a significant drop in their performance in the PointGoal task in Habitat with different datasets and different scenes. Extensive experimental results indicate that there exist serious potential risks for applying Embodied Vision Navigation methods to the real world.



## **40. Test-time Defense against Adversarial Attacks: Detection and Reconstruction of Adversarial Examples via Masked Autoencoder**

cs.CV

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.12848v2) [paper-pdf](http://arxiv.org/pdf/2303.12848v2)

**Authors**: Yun-Yun Tsai, Ju-Chin Chao, Albert Wen, Zhaoyuan Yang, Chengzhi Mao, Tapan Shah, Junfeng Yang

**Abstract**: Existing defense methods against adversarial attacks can be categorized into training time and test time defenses. Training time defense, i.e., adversarial training, requires a significant amount of extra time for training and is often not able to be generalized to unseen attacks. On the other hand, test time defense by test time weight adaptation requires access to perform gradient descent on (part of) the model weights, which could be infeasible for models with frozen weights. To address these challenges, we propose DRAM, a novel defense method to Detect and Reconstruct multiple types of Adversarial attacks via Masked autoencoder (MAE). We demonstrate how to use MAE losses to build a KS-test to detect adversarial attacks. Moreover, the MAE losses can be used to repair adversarial samples from unseen attack types. In this sense, DRAM neither requires model weight updates in test time nor augments the training set with more adversarial samples. Evaluating DRAM on the large-scale ImageNet data, we achieve the best detection rate of 82% on average on eight types of adversarial attacks compared with other detection baselines. For reconstruction, DRAM improves the robust accuracy by 6% ~ 41% for Standard ResNet50 and 3% ~ 8% for Robust ResNet50 compared with other self-supervision tasks, such as rotation prediction and contrastive learning.



## **41. WiFi Physical Layer Stays Awake and Responds When it Should Not**

cs.NI

12 pages

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2301.00269v2) [paper-pdf](http://arxiv.org/pdf/2301.00269v2)

**Authors**: Ali Abedi, Haofan Lu, Alex Chen, Charlie Liu, Omid Abari

**Abstract**: WiFi communication should be possible only between devices inside the same network. However, we find that all existing WiFi devices send back acknowledgments (ACK) to even fake packets received from unauthorized WiFi devices outside of their network. Moreover, we find that an unauthorized device can manipulate the power-saving mechanism of WiFi radios and keep them continuously awake by sending specific fake beacon frames to them. Our evaluation of over 5,000 devices from 186 vendors confirms that these are widespread issues. We believe these loopholes cannot be prevented, and hence they create privacy and security concerns. Finally, to show the importance of these issues and their consequences, we implement and demonstrate two attacks where an adversary performs battery drain and WiFi sensing attacks just using a tiny WiFi module which costs less than ten dollars.



## **42. Ensemble-based Blackbox Attacks on Dense Prediction**

cs.CV

CVPR 2023 Accepted

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14304v1) [paper-pdf](http://arxiv.org/pdf/2303.14304v1)

**Authors**: Zikui Cai, Yaoteng Tan, M. Salman Asif

**Abstract**: We propose an approach for adversarial attacks on dense prediction models (such as object detectors and segmentation). It is well known that the attacks generated by a single surrogate model do not transfer to arbitrary (blackbox) victim models. Furthermore, targeted attacks are often more challenging than the untargeted attacks. In this paper, we show that a carefully designed ensemble can create effective attacks for a number of victim models. In particular, we show that normalization of the weights for individual models plays a critical role in the success of the attacks. We then demonstrate that by adjusting the weights of the ensemble according to the victim model can further improve the performance of the attacks. We performed a number of experiments for object detectors and segmentation to highlight the significance of the our proposed methods. Our proposed ensemble-based method outperforms existing blackbox attack methods for object detection and segmentation. Finally we show that our proposed method can also generate a single perturbation that can fool multiple blackbox detection and segmentation models simultaneously. Code is available at https://github.com/CSIPlab/EBAD.



## **43. Utilizing Network Properties to Detect Erroneous Inputs**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2002.12520v3) [paper-pdf](http://arxiv.org/pdf/2002.12520v3)

**Authors**: Matt Gorbett, Nathaniel Blanchard

**Abstract**: Neural networks are vulnerable to a wide range of erroneous inputs such as adversarial, corrupted, out-of-distribution, and misclassified examples. In this work, we train a linear SVM classifier to detect these four types of erroneous data using hidden and softmax feature vectors of pre-trained neural networks. Our results indicate that these faulty data types generally exhibit linearly separable activation properties from correct examples, giving us the ability to reject bad inputs with no extra training or overhead. We experimentally validate our findings across a diverse range of datasets, domains, pre-trained models, and adversarial attacks.



## **44. How many dimensions are required to find an adversarial example?**

cs.LG

Comments welcome!

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14173v1) [paper-pdf](http://arxiv.org/pdf/2303.14173v1)

**Authors**: Charles Godfrey, Henry Kvinge, Elise Bishoff, Myles Mckay, Davis Brown, Tim Doster, Eleanor Byler

**Abstract**: Past work exploring adversarial vulnerability have focused on situations where an adversary can perturb all dimensions of model input. On the other hand, a range of recent works consider the case where either (i) an adversary can perturb a limited number of input parameters or (ii) a subset of modalities in a multimodal problem. In both of these cases, adversarial examples are effectively constrained to a subspace $V$ in the ambient input space $\mathcal{X}$. Motivated by this, in this work we investigate how adversarial vulnerability depends on $\dim(V)$. In particular, we show that the adversarial success of standard PGD attacks with $\ell^p$ norm constraints behaves like a monotonically increasing function of $\epsilon (\frac{\dim(V)}{\dim \mathcal{X}})^{\frac{1}{q}}$ where $\epsilon$ is the perturbation budget and $\frac{1}{p} + \frac{1}{q} =1$, provided $p > 1$ (the case $p=1$ presents additional subtleties which we analyze in some detail). This functional form can be easily derived from a simple toy linear model, and as such our results land further credence to arguments that adversarial examples are endemic to locally linear models on high dimensional spaces.



## **45. Adversarial Attack and Defense for Medical Image Analysis: Methods and Applications**

eess.IV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14133v1) [paper-pdf](http://arxiv.org/pdf/2303.14133v1)

**Authors**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen

**Abstract**: Deep learning techniques have achieved superior performance in computer-aided medical image analysis, yet they are still vulnerable to imperceptible adversarial attacks, resulting in potential misdiagnosis in clinical practice. Oppositely, recent years have also witnessed remarkable progress in defense against these tailored adversarial examples in deep medical diagnosis systems. In this exposition, we present a comprehensive survey on recent advances in adversarial attack and defense for medical image analysis with a novel taxonomy in terms of the application scenario. We also provide a unified theoretical framework for different types of adversarial attack and defense methods for medical image analysis. For a fair comparison, we establish a new benchmark for adversarially robust medical diagnosis models obtained by adversarial training under various scenarios. To the best of our knowledge, this is the first survey paper that provides a thorough evaluation of adversarially robust medical diagnosis models. By analyzing qualitative and quantitative results, we conclude this survey with a detailed discussion of current challenges for adversarial attack and defense in medical image analysis systems to shed light on future research directions.



## **46. Optimal Smoothing Distribution Exploration for Backdoor Neutralization in Deep Learning-based Traffic Systems**

cs.LG

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14197v1) [paper-pdf](http://arxiv.org/pdf/2303.14197v1)

**Authors**: Yue Wang, Wending Li, Michail Maniatakos, Saif Eddin Jabari

**Abstract**: Deep Reinforcement Learning (DRL) enhances the efficiency of Autonomous Vehicles (AV), but also makes them susceptible to backdoor attacks that can result in traffic congestion or collisions. Backdoor functionality is typically incorporated by contaminating training datasets with covert malicious data to maintain high precision on genuine inputs while inducing the desired (malicious) outputs for specific inputs chosen by adversaries. Current defenses against backdoors mainly focus on image classification using image-based features, which cannot be readily transferred to the regression task of DRL-based AV controllers since the inputs are continuous sensor data, i.e., the combinations of velocity and distance of AV and its surrounding vehicles. Our proposed method adds well-designed noise to the input to neutralize backdoors. The approach involves learning an optimal smoothing (noise) distribution to preserve the normal functionality of genuine inputs while neutralizing backdoors. By doing so, the resulting model is expected to be more resilient against backdoor attacks while maintaining high accuracy on genuine inputs. The effectiveness of the proposed method is verified on a simulated traffic system based on a microscopic traffic simulator, where experimental results showcase that the smoothed traffic controller can neutralize all trigger samples and maintain the performance of relieving traffic congestion



## **47. PIAT: Parameter Interpolation based Adversarial Training for Image Classification**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13955v1) [paper-pdf](http://arxiv.org/pdf/2303.13955v1)

**Authors**: Kun He, Xin Liu, Yichen Yang, Zhou Qin, Weigao Wen, Hui Xue, John E. Hopcroft

**Abstract**: Adversarial training has been demonstrated to be the most effective approach to defend against adversarial attacks. However, existing adversarial training methods show apparent oscillations and overfitting issue in the training process, degrading the defense efficacy. In this work, we propose a novel framework, termed Parameter Interpolation based Adversarial Training (PIAT), that makes full use of the historical information during training. Specifically, at the end of each epoch, PIAT tunes the model parameters as the interpolation of the parameters of the previous and current epochs. Besides, we suggest to use the Normalized Mean Square Error (NMSE) to further improve the robustness by aligning the clean and adversarial examples. Compared with other regularization methods, NMSE focuses more on the relative magnitude of the logits rather than the absolute magnitude. Extensive experiments on several benchmark datasets and various networks show that our method could prominently improve the model robustness and reduce the generalization error. Moreover, our framework is general and could further boost the robust accuracy when combined with other adversarial training methods.



## **48. EC-CFI: Control-Flow Integrity via Code Encryption Counteracting Fault Attacks**

cs.CR

Accepted at HOST'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2301.13760v2) [paper-pdf](http://arxiv.org/pdf/2301.13760v2)

**Authors**: Pascal Nasahl, Salmin Sultana, Hans Liljestrand, Karanvir Grewal, Michael LeMay, David M. Durham, David Schrammel, Stefan Mangard

**Abstract**: Fault attacks enable adversaries to manipulate the control-flow of security-critical applications. By inducing targeted faults into the CPU, the software's call graph can be escaped and the control-flow can be redirected to arbitrary functions inside the program. To protect the control-flow from these attacks, dedicated fault control-flow integrity (CFI) countermeasures are commonly deployed. However, these schemes either have high detection latencies or require intrusive hardware changes. In this paper, we present EC-CFI, a software-based cryptographically enforced CFI scheme with no detection latency utilizing hardware features of recent Intel platforms. Our EC-CFI prototype is designed to prevent an adversary from escaping the program's call graph using faults by encrypting each function with a different key before execution. At runtime, the instrumented program dynamically derives the decryption key, ensuring that the code only can be successfully decrypted when the program follows the intended call graph. To enable this level of protection on Intel commodity systems, we introduce extended page table (EPT) aliasing allowing us to achieve function-granular encryption by combing Intel's TME-MK and virtualization technology. We open-source our custom LLVM-based toolchain automatically protecting arbitrary programs with EC-CFI. Furthermore, we evaluate our EPT aliasing approach with the SPEC CPU2017 and Embench-IoT benchmarks and discuss and evaluate potential TME-MK hardware changes minimizing runtime overheads.



## **49. SCRAMBLE-CFI: Mitigating Fault-Induced Control-Flow Attacks on OpenTitan**

cs.CR

Accepted at GLSVLSI'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.03711v3) [paper-pdf](http://arxiv.org/pdf/2303.03711v3)

**Authors**: Pascal Nasahl, Stefan Mangard

**Abstract**: Secure elements physically exposed to adversaries are frequently targeted by fault attacks. These attacks can be utilized to hijack the control-flow of software allowing the attacker to bypass security measures, extract sensitive data, or gain full code execution. In this paper, we systematically analyze the threat vector of fault-induced control-flow manipulations on the open-source OpenTitan secure element. Our thorough analysis reveals that current countermeasures of this chip either induce large area overheads or still cannot prevent the attacker from exploiting the identified threats. In this context, we introduce SCRAMBLE-CFI, an encryption-based control-flow integrity scheme utilizing existing hardware features of OpenTitan. SCRAMBLE-CFI confines, with minimal hardware overhead, the impact of fault-induced control-flow attacks by encrypting each function with a different encryption tweak at load-time. At runtime, code only can be successfully decrypted when the correct decryption tweak is active. We open-source our hardware changes and release our LLVM toolchain automatically protecting programs. Our analysis shows that SCRAMBLE-CFI complementarily enhances security guarantees of OpenTitan with a negligible hardware overhead of less than 3.97 % and a runtime overhead of 7.02 % for the Embench-IoT benchmarks.



## **50. Foiling Explanations in Deep Neural Networks**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2211.14860v2) [paper-pdf](http://arxiv.org/pdf/2211.14860v2)

**Authors**: Snir Vitrack Tamam, Raz Lapid, Moshe Sipper

**Abstract**: Deep neural networks (DNNs) have greatly impacted numerous fields over the past decade. Yet despite exhibiting superb performance over many problems, their black-box nature still poses a significant challenge with respect to explainability. Indeed, explainable artificial intelligence (XAI) is crucial in several fields, wherein the answer alone -- sans a reasoning of how said answer was derived -- is of little value. This paper uncovers a troubling property of explanation methods for image-based DNNs: by making small visual changes to the input image -- hardly influencing the network's output -- we demonstrate how explanations may be arbitrarily manipulated through the use of evolution strategies. Our novel algorithm, AttaXAI, a model-agnostic, adversarial attack on XAI algorithms, only requires access to the output logits of a classifier and to the explanation map; these weak assumptions render our approach highly useful where real-world models and data are concerned. We compare our method's performance on two benchmark datasets -- CIFAR100 and ImageNet -- using four different pretrained deep-learning models: VGG16-CIFAR100, VGG16-ImageNet, MobileNet-CIFAR100, and Inception-v3-ImageNet. We find that the XAI methods can be manipulated without the use of gradients or other model internals. Our novel algorithm is successfully able to manipulate an image in a manner imperceptible to the human eye, such that the XAI method outputs a specific explanation map. To our knowledge, this is the first such method in a black-box setting, and we believe it has significant value where explainability is desired, required, or legally mandatory.



