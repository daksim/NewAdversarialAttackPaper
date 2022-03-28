# Latest Adversarial Attack Papers
**update at 2022-03-29 06:31:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Origins of Low-dimensional Adversarial Perturbations**

stat.ML

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13779v1)

**Authors**: Elvis Dohmatob, Chuan Guo, Morgane Goibert

**Abstracts**: In this note, we initiate a rigorous study of the phenomenon of low-dimensional adversarial perturbations in classification. These are adversarial perturbations wherein, unlike the classical setting, the attacker's search is limited to a low-dimensional subspace of the feature space. The goal is to fool the classifier into flipping its decision on a nonzero fraction of inputs from a designated class, upon the addition of perturbations from a subspace chosen by the attacker and fixed once and for all. It is desirable that the dimension $k$ of the subspace be much smaller than the dimension $d$ of the feature space, while the norm of the perturbations should be negligible compared to the norm of a typical data point. In this work, we consider binary classification models under very general regularity conditions, which are verified by certain feedforward neural networks (e.g., with sufficiently smooth, or else ReLU activation function), and compute analytical lower-bounds for the fooling rate of any subspace. These bounds explicitly highlight the dependence that the fooling rate has on the margin of the model (i.e., the ratio of the output to its $L_2$-norm of its gradient at a test point), and on the alignment of the given subspace with the gradients of the model w.r.t. inputs. Our results provide a theoretical explanation for the recent success of heuristic methods for efficiently generating low-dimensional adversarial perturbations. Moreover, our theoretical results are confirmed by experiments.



## **2. Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**

cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.05154v2)

**Authors**: Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song

**Abstracts**: Defense models against adversarial attacks have grown significantly, but the lack of practical evaluation methods has hindered progress. Evaluation can be defined as looking for defense models' lower bound of robustness given a budget number of iterations and a test dataset. A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free Adaptive Auto Attack (A$^3$) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. Specifically, by observing that adversarial examples to a specific defense model follow some regularities in their starting points, we design an Adaptive Direction Initialization strategy to speed up the evaluation. Furthermore, to approach the lower bound of robustness under the budget number of iterations, we propose an online statistics-based discarding strategy that automatically identifies and abandons hard-to-attack images. Extensive experiments demonstrate the effectiveness of our A$^3$. Particularly, we apply A$^3$ to nearly 50 widely-used defense models. By consuming much fewer iterations than existing methods, i.e., $1/10$ on average (10$\times$ speed up), we achieve lower robust accuracy in all cases. Notably, we won $\textbf{first place}$ out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method. Code is available at: $\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$



## **3. Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training**

cs.LG

IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR  2022. Preliminary version ICML 2021 Adversarial Machine Learning Workshop.  Code: https://github.com/TheoT1/FW-AT-Adapt

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2012.12368v5)

**Authors**: Theodoros Tsiligkaridis, Jay Roberts

**Abstracts**: Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense. Due to the high computation time for generating strong adversarial examples in the AT process, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training, and although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical framework for adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the $\ell_2$ distortion of $\ell_\infty$ FW attacks. We analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that $\ell_\infty$ attacks against robust models achieve near maximal distortion, while standard networks have lower distortion. It is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. This mathematical transparency differentiates FW from Projected Gradient Descent (PGD) optimization. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps during training to increase efficiency without compromising robustness. FW-AT-Adapt provides training time on par with single-step fast AT methods and closes the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.



## **4. Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness**

cs.CV

to be published in IEEE/CVF Conference on Computer Vision and Pattern  Recognition 2022, CVPR22

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13639v1)

**Authors**: Giulio Lovisotto, Nicole Finnie, Mauricio Munoz, Chaithanya Kumar Mummadi, Jan Hendrik Metzen

**Abstracts**: Neural architectures based on attention such as vision transformers are revolutionizing image recognition. Their main benefit is that attention allows reasoning about all parts of a scene jointly. In this paper, we show how the global reasoning of (scaled) dot-product attention can be the source of a major vulnerability when confronted with adversarial patch attacks. We provide a theoretical understanding of this vulnerability and relate it to an adversary's ability to misdirect the attention of all queries to a single key token under the control of the adversarial patch. We propose novel adversarial objectives for crafting adversarial patches which target this vulnerability explicitly. We show the effectiveness of the proposed patch attacks on popular image classification (ViTs and DeiTs) and object detection models (DETR). We find that adversarial patches occupying 0.5% of the input can lead to robust accuracies as low as 0% for ViT on ImageNet, and reduce the mAP of DETR on MS COCO to less than 3%.



## **5. Adversarial Bone Length Attack on Action Recognition**

cs.CV

12 pages, 8 figures, accepted to AAAI2022

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2109.05830v2)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: Skeleton-based action recognition models have recently been shown to be vulnerable to adversarial attacks. Compared to adversarial attacks on images, perturbations to skeletons are typically bounded to a lower dimension of approximately 100 per frame. This lower-dimensional setting makes it more difficult to generate imperceptible perturbations. Existing attacks resolve this by exploiting the temporal structure of the skeleton motion so that the perturbation dimension increases to thousands. In this paper, we show that adversarial attacks can be performed on skeleton-based action recognition models, even in a significantly low-dimensional setting without any temporal manipulation. Specifically, we restrict the perturbations to the lengths of the skeleton's bones, which allows an adversary to manipulate only approximately 30 effective dimensions. We conducted experiments on the NTU RGB+D and HDM05 datasets and demonstrate that the proposed attack successfully deceived models with sometimes greater than 90% success rate by small perturbations. Furthermore, we discovered an interesting phenomenon: in our low-dimensional setting, the adversarial training with the bone length attack shares a similar property with data augmentation, and it not only improves the adversarial robustness but also improves the classification accuracy on the original data. This is an interesting counterexample of the trade-off between adversarial robustness and clean accuracy, which has been widely observed in studies on adversarial training in the high-dimensional regime.



## **6. Improving Adversarial Transferability with Spatial Momentum**

cs.CV

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13479v1)

**Authors**: Guoqiu Wang, Xingxing Wei, Huanqian Yan

**Abstracts**: Deep Neural Networks (DNN) are vulnerable to adversarial examples. Although many adversarial attack methods achieve satisfactory attack success rates under the white-box setting, they usually show poor transferability when attacking other DNN models. Momentum-based attack (MI-FGSM) is one effective method to improve transferability. It integrates the momentum term into the iterative process, which can stabilize the update directions by adding the gradients' temporal correlation for each pixel. We argue that only this temporal momentum is not enough, the gradients from the spatial domain within an image, i.e. gradients from the context pixels centered on the target pixel are also important to the stabilization. For that, in this paper, we propose a novel method named Spatial Momentum Iterative FGSM Attack (SMI-FGSM), which introduces the mechanism of momentum accumulation from temporal domain to spatial domain by considering the context gradient information from different regions within the image. SMI-FGSM is then integrated with MI-FGSM to simultaneously stabilize the gradients' update direction from both the temporal and spatial domain. The final method is called SM$^2$I-FGSM. Extensive experiments are conducted on the ImageNet dataset and results show that SM$^2$I-FGSM indeed further enhances the transferability. It achieves the best transferability success rate for multiple mainstream undefended and defended models, which outperforms the state-of-the-art methods by a large margin.



## **7. Bounding Training Data Reconstruction in Private (Deep) Learning**

cs.LG

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2201.12383v2)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.



## **8. A Perturbation Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow**

cs.CV

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2203.13214v1)

**Authors**: Jenny Schmalfuss, Philipp Scholze, Andrés Bruhn

**Abstracts**: Recent optical flow methods are almost exclusively judged in terms of accuracy, while analyzing their robustness is often neglected. Although adversarial attacks offer a useful tool to perform such an analysis, current attacks on optical flow methods rather focus on real-world attacking scenarios than on a worst case robustness assessment. Hence, in this work, we propose a novel adversarial attack - the Perturbation Constrained Flow Attack (PCFA) - that emphasizes destructivity over applicability as a real-world attack. More precisely, PCFA is a global attack that optimizes adversarial perturbations to shift the predicted flow towards a specified target flow, while keeping the L2 norm of the perturbation below a chosen bound. Our experiments not only demonstrate PCFA's applicability in white- and black-box settings, but also show that it finds stronger adversarial samples for optical flow than previous attacking frameworks. Moreover, based on these strong samples, we provide the first common ranking of optical flow methods in the literature considering both prediction quality and adversarial robustness, indicating that high quality methods are not necessarily robust. Our source code will be publicly available.



## **9. Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity**

cs.CV

CVPR 2022 conference (accepted), 18 pages, 17 figure

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2203.05151v4)

**Authors**: Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, Linlin Shen

**Abstracts**: Current adversarial attack research reveals the vulnerability of learning-based classifiers against carefully crafted perturbations. However, most existing attack methods have inherent limitations in cross-dataset generalization as they rely on a classification layer with a closed set of categories. Furthermore, the perturbations generated by these methods may appear in regions easily perceptible to the human visual system (HVS). To circumvent the former problem, we propose a novel algorithm that attacks semantic similarity on feature representations. In this way, we are able to fool classifiers without limiting attacks to a specific dataset. For imperceptibility, we introduce the low-frequency constraint to limit perturbations within high-frequency components, ensuring perceptual similarity between adversarial examples and originals. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) and three public online platforms indicate that our attack can yield misleading and transferable adversarial examples across architectures and datasets. Additionally, visualization results and quantitative performance (in terms of four different metrics) show that the proposed algorithm generates more imperceptible perturbations than the state-of-the-art methods. Code is made available at.



## **10. Imperceptible Transfer Attack and Defense on 3D Point Cloud Classification**

cs.CV

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2111.10990v2)

**Authors**: Daizong Liu, Wei Hu

**Abstracts**: Although many efforts have been made into attack and defense on the 2D image domain in recent years, few methods explore the vulnerability of 3D models. Existing 3D attackers generally perform point-wise perturbation over point clouds, resulting in deformed structures or outliers, which is easily perceivable by humans. Moreover, their adversarial examples are generated under the white-box setting, which frequently suffers from low success rates when transferred to attack remote black-box models. In this paper, we study 3D point cloud attacks from two new and challenging perspectives by proposing a novel Imperceptible Transfer Attack (ITA): 1) Imperceptibility: we constrain the perturbation direction of each point along its normal vector of the neighborhood surface, leading to generated examples with similar geometric properties and thus enhancing the imperceptibility. 2) Transferability: we develop an adversarial transformation model to generate the most harmful distortions and enforce the adversarial examples to resist it, improving their transferability to unknown black-box models. Further, we propose to train more robust black-box 3D models to defend against such ITA attacks by learning more discriminative point cloud representations. Extensive evaluations demonstrate that our ITA attack is more imperceptible and transferable than state-of-the-arts and validate the superiority of our defense strategy.



## **11. Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation**

cs.CL

AAAI 2022

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12709v1)

**Authors**: Hanjie Chen, Yangfeng Ji

**Abstracts**: Neural language models show vulnerability to adversarial examples which are semantically similar to their original counterparts with a few words replaced by their synonyms. A common way to improve model robustness is adversarial training which follows two steps-collecting adversarial examples by attacking a target model, and fine-tuning the model on the augmented dataset with these adversarial examples. The objective of traditional adversarial training is to make a model produce the same correct predictions on an original/adversarial example pair. However, the consistency between model decision-makings on two similar texts is ignored. We argue that a robust model should behave consistently on original/adversarial example pairs, that is making the same predictions (what) based on the same reasons (how) which can be reflected by consistent interpretations. In this work, we propose a novel feature-level adversarial training method named FLAT. FLAT aims at improving model robustness in terms of both predictions and interpretations. FLAT incorporates variational word masks in neural networks to learn global word importance and play as a bottleneck teaching the model to make predictions based on important words. FLAT explicitly shoots at the vulnerability problem caused by the mismatch between model understandings on the replaced words and their synonyms in original/adversarial example pairs by regularizing the corresponding global word importance scores. Experiments show the effectiveness of FLAT in improving the robustness with respect to both predictions and interpretations of four neural network models (LSTM, CNN, BERT, and DeBERTa) to two adversarial attacks on four text classification tasks. The models trained via FLAT also show better robustness than baseline models on unforeseen adversarial examples across different attacks.



## **12. Enhancing Classifier Conservativeness and Robustness by Polynomiality**

cs.LG

IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12693v1)

**Authors**: Ziqi Wang, Marco Loog

**Abstracts**: We illustrate the detrimental effect, such as overconfident decisions, that exponential behavior can have in methods like classical LDA and logistic regression. We then show how polynomiality can remedy the situation. This, among others, leads purposefully to random-level performance in the tails, away from the bulk of the training data. A directly related, simple, yet important technical novelty we subsequently present is softRmax: a reasoned alternative to the standard softmax function employed in contemporary (deep) neural networks. It is derived through linking the standard softmax to Gaussian class-conditional models, as employed in LDA, and replacing those by a polynomial alternative. We show that two aspects of softRmax, conservativeness and inherent gradient regularization, lead to robustness against adversarial attacks without gradient obfuscation.



## **13. Explainability-Aware One Point Attack for Point Cloud Neural Networks**

cs.CV

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2110.04158v3)

**Authors**: Hanxiao Tan, Helena Kotthaus

**Abstracts**: With the proposition of neural networks for point clouds, deep learning has started to shine in the field of 3D object recognition while researchers have shown an increased interest to investigate the reliability of point cloud networks by adversarial attacks. However, most of the existing studies aim to deceive humans or defense algorithms, while the few that address the operation principles of the models themselves remain flawed in terms of critical point selection. In this work, we propose two adversarial methods: One Point Attack (OPA) and Critical Traversal Attack (CTA), which incorporate the explainability technologies and aim to explore the intrinsic operating principle of point cloud networks and their sensitivity against critical points perturbations. Our results show that popular point cloud networks can be deceived with almost $100\%$ success rate by shifting only one point from the input instance. In addition, we show the interesting impact of different point attribution distributions on the adversarial robustness of point cloud networks. Finally, we discuss how our approaches facilitate the explainability study for point cloud networks. To the best of our knowledge, this is the first point-cloud-based adversarial approach concerning explainability. Our code is available at https://github.com/Explain3D/Exp-One-Point-Atk-PC.



## **14. Adversarial Fine-tuning for Backdoor Defense: Connecting Backdoor Attacks to Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2202.06312v2)

**Authors**: Bingxu Mu, Zhenxing Niu, Le Wang, Xue Wang, Rong Jin, Gang Hua

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to both backdoor attacks as well as adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, in this paper we find an intriguing connection between them: for a model planted with backdoors, we observe that its adversarial examples have similar behaviors as its triggered samples, i.e., both activate the same subset of DNN neurons. It indicates that planting a backdoor into a model will significantly affect the model's adversarial examples. Based on this observations, we design a new Adversarial Fine-Tuning (AFT) algorithm to defend against backdoor attacks. We empirically show that, against 5 state-of-the-art backdoor attacks, our AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples and significantly outperforms existing defense methods.



## **15. Input-specific Attention Subnetworks for Adversarial Detection**

cs.CL

Accepted at Findings of ACL 2022, 14 pages, 6 Tables and 9 Figures

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12298v1)

**Authors**: Emil Biju, Anirudh Sriram, Pratyush Kumar, Mitesh M Khapra

**Abstracts**: Self-attention heads are characteristic of Transformer models and have been well studied for interpretability and pruning. In this work, we demonstrate an altogether different utility of attention heads, namely for adversarial detection. Specifically, we propose a method to construct input-specific attention subnetworks (IAS) from which we extract three features to discriminate between authentic and adversarial inputs. The resultant detector significantly improves (by over 7.5%) the state-of-the-art adversarial detection accuracy for the BERT encoder on 10 NLU datasets with 11 different adversarial attack types. We also demonstrate that our method (a) is more accurate for larger models which are likely to have more spurious correlations and thus vulnerable to adversarial attack, and (b) performs well even with modest training sets of adversarial examples.



## **16. Integrity Fingerprinting of DNN with Double Black-box Design and Verification**

cs.CR

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.10902v2)

**Authors**: Shuo Wang, Sharif Abuadbba, Sidharth Agarwal, Kristen Moore, Surya Nepal, Salil Kanhere

**Abstracts**: Cloud-enabled Machine Learning as a Service (MLaaS) has shown enormous promise to transform how deep learning models are developed and deployed. Nonetheless, there is a potential risk associated with the use of such services since a malicious party can modify them to achieve an adverse result. Therefore, it is imperative for model owners, service providers, and end-users to verify whether the deployed model has not been tampered with or not. Such verification requires public verifiability (i.e., fingerprinting patterns are available to all parties, including adversaries) and black-box access to the deployed model via APIs. Existing watermarking and fingerprinting approaches, however, require white-box knowledge (such as gradient) to design the fingerprinting and only support private verifiability, i.e., verification by an honest party.   In this paper, we describe a practical watermarking technique that enables black-box knowledge in fingerprint design and black-box queries during verification. The service ensures the integrity of cloud-based services through public verification (i.e. fingerprinting patterns are available to all parties, including adversaries). If an adversary manipulates a model, this will result in a shift in the decision boundary. Thus, the underlying principle of double-black watermarking is that a model's decision boundary could serve as an inherent fingerprint for watermarking. Our approach captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model's decision boundary in order to capture the inherent fingerprints of the model. We evaluated our watermarking approach against a variety of model integrity attacks and model compression attacks.



## **17. Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**

cs.CV

This paper has been accepted by CVPR2022. Code:  https://github.com/hncszyq/ShadowAttack

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.03818v3)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstracts**: Estimating the risk level of adversarial examples is essential for safely deploying machine learning models in the real world. One popular approach for physical-world attacks is to adopt the "sticker-pasting" strategy, which however suffers from some limitations, including difficulties in access to the target or printing by valid colors. A new type of non-invasive attacks emerged recently, which attempt to cast perturbation onto the target by optics based tools, such as laser beam and projector. However, the added optical patterns are artificial but not natural. Thus, they are still conspicuous and attention-grabbed, and can be easily noticed by humans. In this paper, we study a new type of optical adversarial examples, in which the perturbations are generated by a very common natural phenomenon, shadow, to achieve naturalistic and stealthy physical-world adversarial attack under the black-box setting. We extensively evaluate the effectiveness of this new attack on both simulated and real-world environments. Experimental results on traffic sign recognition demonstrate that our algorithm can generate adversarial examples effectively, reaching 98.23% and 90.47% success rates on LISA and GTSRB test sets respectively, while continuously misleading a moving camera over 95% of the time in real-world scenarios. We also offer discussions about the limitations and the defense mechanism of this attack.



## **18. Online Adversarial Attacks**

cs.LG

ICLR 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2103.02014v4)

**Authors**: Andjela Mladenovic, Avishek Joey Bose, Hugo Berard, William L. Hamilton, Simon Lacoste-Julien, Pascal Vincent, Gauthier Gidel

**Abstracts**: Adversarial attacks expose important vulnerabilities of deep learning models, yet little attention has been paid to settings where data arrives as a stream. In this paper, we formalize the online adversarial attack problem, emphasizing two key elements found in real-world use-cases: attackers must operate under partial knowledge of the target model, and the decisions made by the attacker are irrevocable since they operate on a transient data stream. We first rigorously analyze a deterministic variant of the online threat model by drawing parallels to the well-studied $k$-secretary problem in theoretical computer science and propose Virtual+, a simple yet practical online algorithm. Our main theoretical result shows Virtual+ yields provably the best competitive ratio over all single-threshold algorithms for $k<5$ -- extending the previous analysis of the $k$-secretary problem. We also introduce the \textit{stochastic $k$-secretary} -- effectively reducing online blackbox transfer attacks to a $k$-secretary problem under noise -- and prove theoretical bounds on the performance of Virtual+ adapted to this setting. Finally, we complement our theoretical results by conducting experiments on MNIST, CIFAR-10, and Imagenet classifiers, revealing the necessity of online algorithms in achieving near-optimal performance and also the rich interplay between attack strategies and online attack selection, enabling simple strategies like FGSM to outperform stronger adversaries.



## **19. NNReArch: A Tensor Program Scheduling Framework Against Neural Network Architecture Reverse Engineering**

cs.CR

Accepted by FCCM 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.12046v1)

**Authors**: Yukui Luo, Shijin Duan, Cheng Gongye, Yunsi Fei, Xiaolin Xu

**Abstracts**: Architecture reverse engineering has become an emerging attack against deep neural network (DNN) implementations. Several prior works have utilized side-channel leakage to recover the model architecture while the target is executing on a hardware acceleration platform. In this work, we target an open-source deep-learning accelerator, Versatile Tensor Accelerator (VTA), and utilize electromagnetic (EM) side-channel leakage to comprehensively learn the association between DNN architecture configurations and EM emanations. We also consider the holistic system -- including the low-level tensor program code of the VTA accelerator on a Xilinx FPGA and explore the effect of such low-level configurations on the EM leakage. Our study demonstrates that both the optimization and configuration of tensor programs will affect the EM side-channel leakage.   Gaining knowledge of the association between the low-level tensor program and the EM emanations, we propose NNReArch, a lightweight tensor program scheduling framework against side-channel-based DNN model architecture reverse engineering. Specifically, NNReArch targets reshaping the EM traces of different DNN operators, through scheduling the tensor program execution of the DNN model so as to confuse the adversary. NNReArch is a comprehensive protection framework supporting two modes, a balanced mode that strikes a balance between the DNN model confidentiality and execution performance, and a secure mode where the most secure setting is chosen. We implement and evaluate the proposed framework on the open-source VTA with state-of-the-art DNN architectures. The experimental results demonstrate that NNReArch can efficiently enhance the model architecture security with a small performance overhead. In addition, the proposed obfuscation technique makes reverse engineering of the DNN architecture significantly harder.



## **20. Shape-invariant 3D Adversarial Point Clouds**

cs.CV

Accepted at CVPR 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.04041v2)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Nenghai Yu

**Abstracts**: Adversary and invisibility are two fundamental but conflict characters of adversarial perturbations. Previous adversarial attacks on 3D point cloud recognition have often been criticized for their noticeable point outliers, since they just involve an "implicit constrain" like global distance loss in the time-consuming optimization to limit the generated noise. While point cloud is a highly structured data format, it is hard to constrain its perturbation with a simple loss or metric properly. In this paper, we propose a novel Point-Cloud Sensitivity Map to boost both the efficiency and imperceptibility of point perturbations. This map reveals the vulnerability of point cloud recognition models when encountering shape-invariant adversarial noises. These noises are designed along the shape surface with an "explicit constrain" instead of extra distance loss. Specifically, we first apply a reversible coordinate transformation on each point of the point cloud input, to reduce one degree of point freedom and limit its movement on the tangent plane. Then we calculate the best attacking direction with the gradients of the transformed point cloud obtained on the white-box model. Finally we assign each point with a non-negative score to construct the sensitivity map, which benefits both white-box adversarial invisibility and black-box query-efficiency extended in our work. Extensive evaluations prove that our method can achieve the superior performance on various point cloud recognition models, with its satisfying adversarial imperceptibility and strong resistance to different point cloud defense settings. Our code is available at: https://github.com/shikiw/SI-Adv.



## **21. Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis**

cs.LG

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11633v1)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstracts**: Model poisoning attacks on federated learning (FL) intrude in the entire system via compromising an edge model, resulting in malfunctioning of machine learning models. Such compromised models are tampered with to perform adversary-desired behaviors. In particular, we considered a semi-targeted situation where the source class is predetermined however the target class is not. The goal is to cause the global classifier to misclassify data of the source class. Though approaches such as label flipping have been adopted to inject poisoned parameters into FL, it has been shown that their performances are usually class-sensitive varying with different target classes applied. Typically, an attack can become less effective when shifting to a different target class. To overcome this challenge, we propose the Attacking Distance-aware Attack (ADA) to enhance a poisoning attack by finding the optimized target class in the feature space. Moreover, we studied a more challenging situation where an adversary had limited prior knowledge about a client's data. To tackle this problem, ADA deduces pair-wise distances between different classes in the latent feature space from shared model parameters based on the backward error analysis. We performed extensive empirical evaluations on ADA by varying the factor of attacking frequency in three different image classification tasks. As a result, ADA succeeded in increasing the attack performance by 1.8 times in the most challenging case with an attacking frequency of 0.01.



## **22. Exploring High-Order Structure for Robust Graph Structure Learning**

cs.LG

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11492v1)

**Authors**: Guangqian Yang, Yibing Zhan, Jinlong Li, Baosheng Yu, Liu Liu, Fengxiang He

**Abstracts**: Recent studies show that Graph Neural Networks (GNNs) are vulnerable to adversarial attack, i.e., an imperceptible structure perturbation can fool GNNs to make wrong predictions. Some researches explore specific properties of clean graphs such as the feature smoothness to defense the attack, but the analysis of it has not been well-studied. In this paper, we analyze the adversarial attack on graphs from the perspective of feature smoothness which further contributes to an efficient new adversarial defensive algorithm for GNNs. We discover that the effect of the high-order graph structure is a smoother filter for processing graph structures. Intuitively, the high-order graph structure denotes the path number between nodes, where larger number indicates closer connection, so it naturally contributes to defense the adversarial perturbation. Further, we propose a novel algorithm that incorporates the high-order structural information into the graph structure learning. We perform experiments on three popular benchmark datasets, Cora, Citeseer and Polblogs. Extensive experiments demonstrate the effectiveness of our method for defending against graph adversarial attacks.



## **23. Making DeepFakes more spurious: evading deep face forgery detection via trace removal attack**

cs.CV

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11433v1)

**Authors**: Chi Liu, Huajie Chen, Tianqing Zhu, Jun Zhang, Wanlei Zhou

**Abstracts**: DeepFakes are raising significant social concerns. Although various DeepFake detectors have been developed as forensic countermeasures, these detectors are still vulnerable to attacks. Recently, a few attacks, principally adversarial attacks, have succeeded in cloaking DeepFake images to evade detection. However, these attacks have typical detector-specific designs, which require prior knowledge about the detector, leading to poor transferability. Moreover, these attacks only consider simple security scenarios. Less is known about how effective they are in high-level scenarios where either the detectors or the attacker's knowledge varies. In this paper, we solve the above challenges with presenting a novel detector-agnostic trace removal attack for DeepFake anti-forensics. Instead of investigating the detector side, our attack looks into the original DeepFake creation pipeline, attempting to remove all detectable natural DeepFake traces to render the fake images more "authentic". To implement this attack, first, we perform a DeepFake trace discovery, identifying three discernible traces. Then a trace removal network (TR-Net) is proposed based on an adversarial learning framework involving one generator and multiple discriminators. Each discriminator is responsible for one individual trace representation to avoid cross-trace interference. These discriminators are arranged in parallel, which prompts the generator to remove various traces simultaneously. To evaluate the attack efficacy, we crafted heterogeneous security scenarios where the detectors were embedded with different levels of defense and the attackers' background knowledge of data varies. The experimental results show that the proposed attack can significantly compromise the detection accuracy of six state-of-the-art DeepFake detectors while causing only a negligible loss in visual quality to the original DeepFake samples.



## **24. Subspace Adversarial Training**

cs.LG

CVPR2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2111.12229v2)

**Authors**: Tao Li, Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: Single-step adversarial training (AT) has received wide attention as it proved to be both efficient and robust. However, a serious problem of catastrophic overfitting exists, i.e., the robust accuracy against projected gradient descent (PGD) attack suddenly drops to 0% during the training. In this paper, we approach this problem from a novel perspective of optimization and firstly reveal the close link between the fast-growing gradient of each sample and overfitting, which can also be applied to understand robust overfitting in multi-step AT. To control the growth of the gradient, we propose a new AT method, Subspace Adversarial Training (Sub-AT), which constrains AT in a carefully extracted subspace. It successfully resolves both kinds of overfitting and significantly boosts the robustness. In subspace, we also allow single-step AT with larger steps and larger radius, further improving the robustness performance. As a result, we achieve state-of-the-art single-step AT performance. Without any regularization term, our single-step AT can reach over 51% robust accuracy against strong PGD-50 attack of radius 8/255 on CIFAR-10, reaching a competitive performance against standard multi-step PGD-10 AT with huge computational advantages. The code is released at https://github.com/nblt/Sub-AT.



## **25. On The Robustness of Offensive Language Classifiers**

cs.CL

9 pages, 2 figures, Accepted at ACL 2022

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.11331v1)

**Authors**: Jonathan Rusert, Zubair Shafiq, Padmini Srinivasan

**Abstracts**: Social media platforms are deploying machine learning based offensive language classification systems to combat hateful, racist, and other forms of offensive speech at scale. However, despite their real-world deployment, we do not yet comprehensively understand the extent to which offensive language classifiers are robust against adversarial attacks. Prior work in this space is limited to studying robustness of offensive language classifiers against primitive attacks such as misspellings and extraneous spaces. To address this gap, we systematically analyze the robustness of state-of-the-art offensive language classifiers against more crafty adversarial attacks that leverage greedy- and attention-based word selection and context-aware embeddings for word replacement. Our results on multiple datasets show that these crafty adversarial attacks can degrade the accuracy of offensive language classifiers by more than 50% while also being able to preserve the readability and meaning of the modified text.



## **26. FGAN: Federated Generative Adversarial Networks for Anomaly Detection in Network Traffic**

cs.CR

8 pages, 2 figures

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.11106v1)

**Authors**: Sankha Das

**Abstracts**: Over the last two decades, a lot of work has been done in improving network security, particularly in intrusion detection systems (IDS) and anomaly detection. Machine learning solutions have also been employed in IDSs to detect known and plausible attacks in incoming traffic. Parameters such as packet contents, sender IP and sender port, connection duration, etc. have been previously used to train these machine learning models to learn to differentiate genuine traffic from malicious ones. Generative Adversarial Networks (GANs) have been significantly successful in detecting such anomalies, mostly attributed to the adversarial training of the generator and discriminator in an attempt to bypass each other and in turn increase their own power and accuracy. However, in large networks having a wide variety of traffic at possibly different regions of the network and susceptible to a large number of potential attacks, training these GANs for a particular kind of anomaly may make it oblivious to other anomalies and attacks. In addition, the dataset required to train these models has to be made centrally available and publicly accessible, posing the obvious question of privacy of the communications of the respective participants of the network. The solution proposed in this work aims at tackling the above two issues by using GANs in a federated architecture in networks of such scale and capacity. In such a setting, different users of the network will be able to train and customize a centrally available adversarial model according to their own frequently faced conditions. Simultaneously, the member users of the network will also able to gain from the experiences of the other users in the network.



## **27. An Intermediate-level Attack Framework on The Basis of Linear Regression**

cs.CV

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10723v1)

**Authors**: Yiwen Guo, Qizhang Li, Wangmeng Zuo, Hao Chen

**Abstracts**: This paper substantially extends our work published at ECCV, in which an intermediate-level attack was proposed to improve the transferability of some baseline adversarial examples. We advocate to establish a direct linear mapping from the intermediate-level discrepancies (between adversarial features and benign features) to classification prediction loss of the adversarial example. In this paper, we delve deep into the core components of such a framework by performing comprehensive studies and extensive experiments. We show that 1) a variety of linear regression models can all be considered in order to establish the mapping, 2) the magnitude of the finally obtained intermediate-level discrepancy is linearly correlated with adversarial transferability, 3) further boost of the performance can be achieved by performing multiple runs of the baseline attack with random initialization. By leveraging these findings, we achieve new state-of-the-arts on transfer-based $\ell_\infty$ and $\ell_2$ attacks.



## **28. A Prompting-based Approach for Adversarial Example Generation and Robustness Enhancement**

cs.CL

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10714v1)

**Authors**: Yuting Yang, Pei Huang, Juan Cao, Jintao Li, Yun Lin, Jin Song Dong, Feifei Ma, Jian Zhang

**Abstracts**: Recent years have seen the wide application of NLP models in crucial areas such as finance, medical treatment, and news media, raising concerns of the model robustness and vulnerabilities. In this paper, we propose a novel prompt-based adversarial attack to compromise NLP models and robustness enhancement technique. We first construct malicious prompts for each instance and generate adversarial examples via mask-and-filling under the effect of a malicious purpose. Our attack technique targets the inherent vulnerabilities of NLP models, allowing us to generate samples even without interacting with the victim NLP model, as long as it is based on pre-trained language models (PLMs). Furthermore, we design a prompt-based adversarial training method to improve the robustness of PLMs. As our training method does not actually generate adversarial samples, it can be applied to large-scale training sets efficiently. The experimental results show that our attack method can achieve a high attack success rate with more diverse, fluent and natural adversarial examples. In addition, our robustness enhancement method can significantly improve the robustness of models to resist adversarial attacks. Our work indicates that prompting paradigm has great potential in probing some fundamental flaws of PLMs and fine-tuning them for downstream tasks.



## **29. Leveraging Expert Guided Adversarial Augmentation For Improving Generalization in Named Entity Recognition**

cs.CL

ACL 2022 (Findings)

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10693v1)

**Authors**: Aaron Reich, Jiaao Chen, Aastha Agrawal, Yanzhe Zhang, Diyi Yang

**Abstracts**: Named Entity Recognition (NER) systems often demonstrate great performance on in-distribution data, but perform poorly on examples drawn from a shifted distribution. One way to evaluate the generalization ability of NER models is to use adversarial examples, on which the specific variations associated with named entities are rarely considered. To this end, we propose leveraging expert-guided heuristics to change the entity tokens and their surrounding contexts thereby altering their entity types as adversarial attacks. Using expert-guided heuristics, we augmented the CoNLL 2003 test set and manually annotated it to construct a high-quality challenging set. We found that state-of-the-art NER systems trained on CoNLL 2003 training data drop performance dramatically on our challenging set. By training on adversarial augmented training examples and using mixup for regularization, we were able to significantly improve the performance on the challenging set as well as improve out-of-domain generalization which we evaluated by using OntoNotes data. We have publicly released our dataset and code at https://github.com/GT-SALT/Guided-Adversarial-Augmentation.



## **30. RareGAN: Generating Samples for Rare Classes**

cs.LG

Published in AAAI 2022

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10674v1)

**Authors**: Zinan Lin, Hao Liang, Giulia Fanti, Vyas Sekar

**Abstracts**: We study the problem of learning generative adversarial networks (GANs) for a rare class of an unlabeled dataset subject to a labeling budget. This problem is motivated from practical applications in domains including security (e.g., synthesizing packets for DNS amplification attacks), systems and networking (e.g., synthesizing workloads that trigger high resource usage), and machine learning (e.g., generating images from a rare class). Existing approaches are unsuitable, either requiring fully-labeled datasets or sacrificing the fidelity of the rare class for that of the common classes. We propose RareGAN, a novel synthesis of three key ideas: (1) extending conditional GANs to use labelled and unlabelled data for better generalization; (2) an active learning approach that requests the most useful labels; and (3) a weighted loss function to favor learning the rare class. We show that RareGAN achieves a better fidelity-diversity tradeoff on the rare class than prior work across different applications, budgets, rare class fractions, GAN losses, and architectures.



## **31. Does DQN really learn? Exploring adversarial training schemes in Pong**

cs.LG

RLDM 2022

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10614v1)

**Authors**: Bowen He, Sreehari Rammohan, Jessica Forde, Michael Littman

**Abstracts**: In this work, we study two self-play training schemes, Chainer and Pool, and show they lead to improved agent performance in Atari Pong compared to a standard DQN agent -- trained against the built-in Atari opponent. To measure agent performance, we define a robustness metric that captures how difficult it is to learn a strategy that beats the agent's learned policy. Through playing past versions of themselves, Chainer and Pool are able to target weaknesses in their policies and improve their resistance to attack. Agents trained using these methods score well on our robustness metric and can easily defeat the standard DQN agent. We conclude by using linear probing to illuminate what internal structures the different agents develop to play the game. We show that training agents with Chainer or Pool leads to richer network activations with greater predictive power to estimate critical game-state features compared to the standard DQN agent.



## **32. Improved Semi-Quantum Key Distribution with Two Almost-Classical Users**

quant-ph

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10567v1)

**Authors**: Saachi Mutreja, Walter O. Krawec

**Abstracts**: Semi-quantum key distribution (SQKD) protocols attempt to establish a shared secret key between users, secure against computationally unbounded adversaries. Unlike standard quantum key distribution protocols, SQKD protocols contain at least one user who is limited in their quantum abilities and is almost "classical" in nature. In this paper, we revisit a mediated semi-quantum key distribution protocol, introduced by Massa et al., in 2019, where users need only the ability to detect a qubit, or reflect a qubit; they do not need to perform any other basis measurement; nor do they need to prepare quantum signals. Users require the services of a quantum server which may be controlled by the adversary. In this paper, we show how this protocol may be extended to improve its efficiency and also its noise tolerance. We discuss an extension which allows more communication rounds to be directly usable; we analyze the key-rate of this extension in the asymptotic scenario for a particular class of attacks and compare with prior work. Finally, we evaluate the protocol's performance in a variety of lossy and noisy channels.



## **33. Adversarial Parameter Attack on Deep Neural Networks**

cs.LG

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10502v1)

**Authors**: Lijia Yu, Yihan Wang, Xiao-Shan Gao

**Abstracts**: In this paper, a new parameter perturbation attack on DNNs, called adversarial parameter attack, is proposed, in which small perturbations to the parameters of the DNN are made such that the accuracy of the attacked DNN does not decrease much, but its robustness becomes much lower. The adversarial parameter attack is stronger than previous parameter perturbation attacks in that the attack is more difficult to be recognized by users and the attacked DNN gives a wrong label for any modified sample input with high probability. The existence of adversarial parameters is proved. For a DNN $F_{\Theta}$ with the parameter set $\Theta$ satisfying certain conditions, it is shown that if the depth of the DNN is sufficiently large, then there exists an adversarial parameter set $\Theta_a$ for $\Theta$ such that the accuracy of $F_{\Theta_a}$ is equal to that of $F_{\Theta}$, but the robustness measure of $F_{\Theta_a}$ is smaller than any given bound. An effective training algorithm is given to compute adversarial parameters and numerical experiments are used to demonstrate that the algorithms are effective to produce high quality adversarial parameters.



## **34. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

cs.LG

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.08959v2)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.



## **35. On Robust Prefix-Tuning for Text Classification**

cs.CL

Accepted in ICLR 2022. We release the code at  https://github.com/minicheshire/Robust-Prefix-Tuning

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10378v1)

**Authors**: Zonghan Yang, Yang Liu

**Abstracts**: Recently, prefix-tuning has gained increasing attention as a parameter-efficient finetuning method for large-scale pretrained language models. The method keeps the pretrained models fixed and only updates the prefix token parameters for each downstream task. Despite being lightweight and modular, prefix-tuning still lacks robustness to textual adversarial attacks. However, most currently developed defense techniques necessitate auxiliary model update and storage, which inevitably hamper the modularity and low storage of prefix-tuning. In this work, we propose a robust prefix-tuning framework that preserves the efficiency and modularity of prefix-tuning. The core idea of our framework is leveraging the layerwise activations of the language model by correctly-classified training data as the standard for additional prefix finetuning. During the test phase, an extra batch-level prefix is tuned for each batch and added to the original prefix for robustness enhancement. Extensive experiments on three text classification benchmarks show that our framework substantially improves robustness over several strong baselines against five textual attacks of different types while maintaining comparable accuracy on clean texts. We also interpret our robust prefix-tuning framework from the optimal control perspective and pose several directions for future research.



## **36. Perturbations in the Wild: Leveraging Human-Written Text Perturbations for Realistic Adversarial Attack and Defense**

cs.LG

Accepted to the 60th Annual Meeting of the Association for  Computational Linguistics (ACL'22), Findings

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10346v1)

**Authors**: Thai Le, Jooyoung Lee, Kevin Yen, Yifan Hu, Dongwon Lee

**Abstracts**: We proposes a novel algorithm, ANTHRO, that inductively extracts over 600K human-written text perturbations in the wild and leverages them for realistic adversarial attack. Unlike existing character-based attacks which often deductively hypothesize a set of manipulation strategies, our work is grounded on actual observations from real-world texts. We find that adversarial texts generated by ANTHRO achieve the best trade-off between (1) attack success rate, (2) semantic preservation of the original text, and (3) stealthiness--i.e. indistinguishable from human writings hence harder to be flagged as suspicious. Specifically, our attacks accomplished around 83% and 91% attack success rates on BERT and RoBERTa, respectively. Moreover, it outperformed the TextBugger baseline with an increase of 50% and 40% in terms of semantic preservation and stealthiness when evaluated by both layperson and professional human workers. ANTHRO can further enhance a BERT classifier's performance in understanding different variations of human-written toxic texts via adversarial training when compared to the Perspective API.



## **37. Efficient Neural Network Analysis with Sum-of-Infeasibilities**

cs.LG

TACAS'22

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.11201v1)

**Authors**: Haoze Wu, Aleksandar Zeljić, Guy Katz, Clark Barrett

**Abstracts**: Inspired by sum-of-infeasibilities methods in convex optimization, we propose a novel procedure for analyzing verification queries on neural networks with piecewise-linear activation functions. Given a convex relaxation which over-approximates the non-convex activation functions, we encode the violations of activation functions as a cost function and optimize it with respect to the convex relaxation. The cost function, referred to as the Sum-of-Infeasibilities (SoI), is designed so that its minimum is zero and achieved only if all the activation functions are satisfied. We propose a stochastic procedure, DeepSoI, to efficiently minimize the SoI. An extension to a canonical case-analysis-based complete search procedure can be achieved by replacing the convex procedure executed at each search state with DeepSoI. Extending the complete search with DeepSoI achieves multiple simultaneous goals: 1) it guides the search towards a counter-example; 2) it enables more informed branching decisions; and 3) it creates additional opportunities for bound derivation. An extensive evaluation across different benchmarks and solvers demonstrates the benefit of the proposed techniques. In particular, we demonstrate that SoI significantly improves the performance of an existing complete search procedure. Moreover, the SoI-based implementation outperforms other state-of-the-art complete verifiers. We also show that our technique can efficiently improve upon the perturbation bound derived by a recent adversarial attack algorithm.



## **38. Distinguishing Non-natural from Natural Adversarial Samples for More Robust Pre-trained Language Model**

cs.LG

Accepted by findings of ACL 2022

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.11199v1)

**Authors**: Jiayi Wang, Rongzhou Bao, Zhuosheng Zhang, Hai Zhao

**Abstracts**: Recently, the problem of robustness of pre-trained language models (PrLMs) has received increasing research interest. Latest studies on adversarial attacks achieve high attack success rates against PrLMs, claiming that PrLMs are not robust. However, we find that the adversarial samples that PrLMs fail are mostly non-natural and do not appear in reality. We question the validity of current evaluation of robustness of PrLMs based on these non-natural adversarial samples and propose an anomaly detector to evaluate the robustness of PrLMs with more natural adversarial samples. We also investigate two applications of the anomaly detector: (1) In data augmentation, we employ the anomaly detector to force generating augmented data that are distinguished as non-natural, which brings larger gains to the accuracy of PrLMs. (2) We apply the anomaly detector to a defense framework to enhance the robustness of PrLMs. It can be used to defend all types of attacks and achieves higher accuracy on both adversarial samples and compliant samples than other defense frameworks.



## **39. Adversarial Defense via Image Denoising with Chaotic Encryption**

cs.LG

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10290v1)

**Authors**: Shi Hu, Eric Nalisnick, Max Welling

**Abstracts**: In the literature on adversarial examples, white box and black box attacks have received the most attention. The adversary is assumed to have either full (white) or no (black) access to the defender's model. In this work, we focus on the equally practical gray box setting, assuming an attacker has partial information. We propose a novel defense that assumes everything but a private key will be made available to the attacker. Our framework uses an image denoising procedure coupled with encryption via a discretized Baker map. Extensive testing against adversarial images (e.g. FGSM, PGD) crafted using various gradients shows that our defense achieves significantly better results on CIFAR-10 and CIFAR-100 than the state-of-the-art gray box defenses in both natural and adversarial accuracy.



## **40. Synthesis of the Supremal Covert Attacker Against Unknown Supervisors by Using Observations**

eess.SY

arXiv admin note: text overlap with arXiv:2106.12268

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.08360v2)

**Authors**: Ruochen Tai, Liyong Lin, Yuting Zhu, Rong Su

**Abstracts**: In this paper, we consider the problem of synthesizing the supremal covert damage-reachable attacker, in the setup where the model of the supervisor is unknown to the adversary but the adversary has recorded a (prefix-closed) finite set of observations of the runs of the closed-loop system. The synthesized attacker needs to ensure both the damage-reachability and the covertness against all the supervisors which are consistent with the given set of observations. There is a gap between the de facto supremality, assuming the model of the supervisor is known, and the supremality that can be attained with a limited knowledge of the model of the supervisor, from the adversary's point of view. We consider the setup where the attacker can exercise sensor replacement/deletion attacks and actuator enablement/disablement attacks. The solution methodology proposed in this work is to reduce the synthesis of the supremal covert damage-reachable attacker, given the model of the plant and the finite set of observations, to the synthesis of the supremal safe supervisor for certain transformed plant, which shows the decidability of the observation-assisted covert attacker synthesis problem. The effectiveness of our approach is illustrated on a water tank example adapted from the literature.



## **41. Adversarial Attacks on Deep Learning-based Video Compression and Classification Systems**

cs.CV

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.10183v1)

**Authors**: Jung-Woo Chang, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstracts**: Video compression plays a crucial role in enabling video streaming and classification systems and maximizing the end-user quality of experience (QoE) at a given bandwidth budget. In this paper, we conduct the first systematic study for adversarial attacks on deep learning based video compression and downstream classification systems. We propose an adaptive adversarial attack that can manipulate the Rate-Distortion (R-D) relationship of a video compression model to achieve two adversarial goals: (1) increasing the network bandwidth or (2) degrading the video quality for end-users. We further devise novel objectives for targeted and untargeted attacks to a downstream video classification service. Finally, we design an input-invariant perturbation that universally disrupts video compression and classification systems in real time. Unlike previously proposed attacks on video classification, our adversarial perturbations are the first to withstand compression. We empirically show the resilience of our attacks against various defenses, i.e., adversarial training, video denoising, and JPEG compression. Our extensive experimental results on various video datasets demonstrate the effectiveness of our attacks. Our video quality and bandwidth attacks deteriorate peak signal-to-noise ratio by up to 5.4dB and the bit-rate by up to 2.4 times on the standard video compression datasets while achieving over 90% attack success rate on a downstream classifier.



## **42. Concept-based Adversarial Attacks: Tricking Humans and Classifiers Alike**

cs.LG

Accepted at IEEE Symposium on Security and Privacy (S&P) Workshop on  Deep Learning and Security, 2022

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.10166v1)

**Authors**: Johannes Schneider, Giovanni Apruzzese

**Abstracts**: We propose to generate adversarial samples by modifying activations of upper layers encoding semantically meaningful concepts. The original sample is shifted towards a target sample, yielding an adversarial sample, by using the modified activations to reconstruct the original sample. A human might (and possibly should) notice differences between the original and the adversarial sample. Depending on the attacker-provided constraints, an adversarial sample can exhibit subtle differences or appear like a "forged" sample from another class. Our approach and goal are in stark contrast to common attacks involving perturbations of single pixels that are not recognizable by humans. Our approach is relevant in, e.g., multi-stage processing of inputs, where both humans and machines are involved in decision-making because invisible perturbations will not fool a human. Our evaluation focuses on deep neural networks. We also show the transferability of our adversarial examples among networks.



## **43. All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines**

cs.CV

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2112.09219v2)

**Authors**: Yuxuan Zhang, Bo Dong, Felix Heide

**Abstracts**: Existing neural networks for computer vision tasks are vulnerable to adversarial attacks: adding imperceptible perturbations to the input images can fool these methods to make a false prediction on an image that was correctly predicted without the perturbation. Various defense methods have proposed image-to-image mapping methods, either including these perturbations in the training process or removing them in a preprocessing denoising step. In doing so, existing methods often ignore that the natural RGB images in today's datasets are not captured but, in fact, recovered from RAW color filter array captures that are subject to various degradations in the capture. In this work, we exploit this RAW data distribution as an empirical prior for adversarial defense. Specifically, we proposed a model-agnostic adversarial defensive method, which maps the input RGB images to Bayer RAW space and back to output RGB using a learned camera image signal processing (ISP) pipeline to eliminate potential adversarial patterns. The proposed method acts as an off-the-shelf preprocessing module and, unlike model-specific adversarial training methods, does not require adversarial images to train. As a result, the method generalizes to unseen tasks without additional retraining. Experiments on large-scale datasets (e.g., ImageNet, COCO) for different vision tasks (e.g., classification, semantic segmentation, object detection) validate that the method significantly outperforms existing methods across task domains.



## **44. Graph-Fraudster: Adversarial Attacks on Graph Neural Network Based Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2110.06468v2)

**Authors**: Jinyin Chen, Guohan Huang, Haibin Zheng, Shanqing Yu, Wenrong Jiang, Chen Cui

**Abstracts**: Graph neural network (GNN) has achieved great success on graph representation learning. Challenged by large scale private data collected from user-side, GNN may not be able to reflect the excellent performance, without rich features and complete adjacent relationships. Addressing the problem, vertical federated learning (VFL) is proposed to implement local data protection through training a global model collaboratively. Consequently, for graph-structured data, it is a natural idea to construct a GNN based VFL framework, denoted as GVFL. However, GNN has been proved vulnerable to adversarial attacks. Whether the vulnerability will be brought into the GVFL has not been studied. This is the first study of adversarial attacks on GVFL. A novel adversarial attack method is proposed, named Graph-Fraudster. It generates adversarial perturbations based on the noise-added global node embeddings via the privacy leakage and the gradient of pairwise node. Specifically, first, Graph-Fraudster steals the global node embeddings and sets up a shadow model of the server for the attack generator. Second, noise is added into node embeddings to confuse the shadow model. At last, the gradient of pairwise node is used to generate attacks with the guidance of noise-added node embeddings. Extensive experiments on five benchmark datasets demonstrate that Graph-Fraudster achieves the state-of-the-art attack performance compared with baselines in different GNN based GVFLs. Furthermore, Graph-Fraudster can remain a threat to GVFL even if two possible defense mechanisms are applied. Additionally, some suggestions are put forward for the future work to improve the robustness of GVFL. The code and datasets can be downloaded at https://github.com/hgh0545/Graph-Fraudster.



## **45. Defending Variational Autoencoders from Adversarial Attacks with MCMC**

cs.LG

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09940v1)

**Authors**: Anna Kuzina, Max Welling, Jakub M. Tomczak

**Abstracts**: Variational autoencoders (VAEs) are deep generative models used in various domains. VAEs can generate complex objects and provide meaningful latent representations, which can be further used in downstream tasks such as classification. As previous work has shown, one can easily fool VAEs to produce unexpected latent representations and reconstructions for a visually slightly modified input. Here, we examine several objective functions for adversarial attacks construction, suggest metrics assess the model robustness, and propose a solution to alleviate the effect of an attack. Our method utilizes the Markov Chain Monte Carlo (MCMC) technique in the inference step and is motivated by our theoretical analysis. Thus, we do not incorporate any additional costs during training or we do not decrease the performance on non-attacked inputs. We validate our approach on a variety of datasets (MNIST, Fashion MNIST, Color MNIST, CelebA) and VAE configurations ($\beta$-VAE, NVAE, TC-VAE) and show that it consistently improves the model robustness to adversarial attacks.



## **46. Neural Predictor for Black-Box Adversarial Attacks on Speech Recognition**

cs.SD

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09849v1)

**Authors**: Marie Biolková, Bac Nguyen

**Abstracts**: Recent works have revealed the vulnerability of automatic speech recognition (ASR) models to adversarial examples (AEs), i.e., small perturbations that cause an error in the transcription of the audio signal. Studying audio adversarial attacks is therefore the first step towards robust ASR. Despite the significant progress made in attacking audio examples, the black-box attack remains challenging because only the hard-label information of transcriptions is provided. Due to this limited information, existing black-box methods often require an excessive number of queries to attack a single audio example. In this paper, we introduce NP-Attack, a neural predictor-based method, which progressively evolves the search towards a small adversarial perturbation. Given a perturbation direction, our neural predictor directly estimates the smallest perturbation that causes a mistranscription. In particular, it enables NP-Attack to accurately learn promising perturbation directions via gradient-based optimization. Experimental results show that NP-Attack achieves competitive results with other state-of-the-art black-box adversarial attacks while requiring a significantly smaller number of queries. The code of NP-Attack is available online.



## **47. DTA: Physical Camouflage Attacks using Differentiable Transformation Network**

cs.CV

Accepted for CVPR 2022

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09831v1)

**Authors**: Naufal Suryanto, Yongsu Kim, Hyoeun Kang, Harashta Tatimma Larasati, Youngyeo Yun, Thi-Thu-Huong Le, Hunmin Yang, Se-Yoon Oh, Howon Kim

**Abstracts**: To perform adversarial attacks in the physical world, many studies have proposed adversarial camouflage, a method to hide a target object by applying camouflage patterns on 3D object surfaces. For obtaining optimal physical adversarial camouflage, previous studies have utilized the so-called neural renderer, as it supports differentiability. However, existing neural renderers cannot fully represent various real-world transformations due to a lack of control of scene parameters compared to the legacy photo-realistic renderers. In this paper, we propose the Differentiable Transformation Attack (DTA), a framework for generating a robust physical adversarial pattern on a target object to camouflage it against object detection models with a wide range of transformations. It utilizes our novel Differentiable Transformation Network (DTN), which learns the expected transformation of a rendered object when the texture is changed while preserving the original properties of the target object. Using our attack framework, an adversary can gain both the advantages of the legacy photo-realistic renderers including various physical-world transformations and the benefit of white-box access by offering differentiability. Our experiments show that our camouflaged 3D vehicles can successfully evade state-of-the-art object detection models in the photo-realistic environment (i.e., CARLA on Unreal Engine). Furthermore, our demonstration on a scaled Tesla Model 3 proves the applicability and transferability of our method to the real world.



## **48. AdIoTack: Quantifying and Refining Resilience of Decision Tree Ensemble Inference Models against Adversarial Volumetric Attacks on IoT Networks**

cs.LG

15 pages, 16 figures, 4 tables

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09792v1)

**Authors**: Arman Pashamokhtari, Gustavo Batista, Hassan Habibi Gharakheili

**Abstracts**: Machine Learning-based techniques have shown success in cyber intelligence. However, they are increasingly becoming targets of sophisticated data-driven adversarial attacks resulting in misprediction, eroding their ability to detect threats on network devices. In this paper, we present AdIoTack, a system that highlights vulnerabilities of decision trees against adversarial attacks, helping cybersecurity teams quantify and refine the resilience of their trained models for monitoring IoT networks. To assess the model for the worst-case scenario, AdIoTack performs white-box adversarial learning to launch successful volumetric attacks that decision tree ensemble models cannot flag. Our first contribution is to develop a white-box algorithm that takes a trained decision tree ensemble model and the profile of an intended network-based attack on a victim class as inputs. It then automatically generates recipes that specify certain packets on top of the indented attack packets (less than 15% overhead) that together can bypass the inference model unnoticed. We ensure that the generated attack instances are feasible for launching on IP networks and effective in their volumetric impact. Our second contribution develops a method to monitor the network behavior of connected devices actively, inject adversarial traffic (when feasible) on behalf of a victim IoT device, and successfully launch the intended attack. Our third contribution prototypes AdIoTack and validates its efficacy on a testbed consisting of a handful of real IoT devices monitored by a trained inference model. We demonstrate how the model detects all non-adversarial volumetric attacks on IoT devices while missing many adversarial ones. The fourth contribution develops systematic methods for applying patches to trained decision tree ensemble models, improving their resilience against adversarial volumetric attacks.



## **49. Adversarial Texture for Fooling Person Detectors in the Physical World**

cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.03373v3)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Xiaolin Hu, Fuchun Sun, Bo Zhang

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.



## **50. AutoAdversary: A Pixel Pruning Method for Sparse Adversarial Attack**

cs.CV

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09756v1)

**Authors**: Jinqiao Li, Xiaotao Liu, Jian Zhao, Furao Shen

**Abstracts**: Deep neural networks (DNNs) have been proven to be vulnerable to adversarial examples. A special branch of adversarial examples, namely sparse adversarial examples, can fool the target DNNs by perturbing only a few pixels. However, many existing sparse adversarial attacks use heuristic methods to select the pixels to be perturbed, and regard the pixel selection and the adversarial attack as two separate steps. From the perspective of neural network pruning, we propose a novel end-to-end sparse adversarial attack method, namely AutoAdversary, which can find the most important pixels automatically by integrating the pixel selection into the adversarial attack. Specifically, our method utilizes a trainable neural network to generate a binary mask for the pixel selection. After jointly optimizing the adversarial perturbation and the neural network, only the pixels corresponding to the value 1 in the mask are perturbed. Experiments demonstrate the superiority of our proposed method over several state-of-the-art methods. Furthermore, since AutoAdversary does not require a heuristic pixel selection process, it does not slow down excessively as other methods when the image size increases.



