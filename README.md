# Latest Adversarial Attack Papers
**update at 2022-03-30 06:31:59**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Poisoning and Backdooring Contrastive Learning**

cs.LG

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2106.09667v2)

**Authors**: Nicholas Carlini, Andreas Terzis

**Abstracts**: Multimodal contrastive learning methods like CLIP train on noisy and uncurated training datasets. This is cheaper than labeling datasets manually, and even improves out-of-distribution robustness. We show that this practice makes backdoor and poisoning attacks a significant threat. By poisoning just 0.01% of a dataset (e.g., just 300 images of the 3 million-example Conceptual Captions dataset), we can cause the model to misclassify test images by overlaying a small patch. Targeted poisoning attacks, whereby the model misclassifies a particular test input with an adversarially-desired label, are even easier requiring control of 0.0001% of the dataset (e.g., just three out of the 3 million images). Our attacks call into question whether training on noisy and uncurated Internet scrapes is desirable.



## **2. Boosting Black-Box Adversarial Attacks with Meta Learning**

cs.LG

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.14607v1)

**Authors**: Junjie Fu, Jian Sun, Gang Wang

**Abstracts**: Deep neural networks (DNNs) have achieved remarkable success in diverse fields. However, it has been demonstrated that DNNs are very vulnerable to adversarial examples even in black-box settings. A large number of black-box attack methods have been proposed to in the literature. However, those methods usually suffer from low success rates and large query counts, which cannot fully satisfy practical purposes. In this paper, we propose a hybrid attack method which trains meta adversarial perturbations (MAPs) on surrogate models and performs black-box attacks by estimating gradients of the models. Our method uses the meta adversarial perturbation as an initialization and subsequently trains any black-box attack method for several epochs. Furthermore, the MAPs enjoy favorable transferability and universality, in the sense that they can be employed to boost performance of other black-box adversarial attack methods. Extensive experiments demonstrate that our method can not only improve the attack success rates, but also reduces the number of queries compared to other methods.



## **3. Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**

cs.CV

Accepted by CVPR2022. Code is available at  https://github.com/CGCL-codes/AMT-GAN

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.03121v2)

**Authors**: Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, Libing Wu

**Abstracts**: While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality. In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.



## **4. Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**

cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.05154v3)

**Authors**: Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song

**Abstracts**: Defense models against adversarial attacks have grown significantly, but the lack of practical evaluation methods has hindered progress. Evaluation can be defined as looking for defense models' lower bound of robustness given a budget number of iterations and a test dataset. A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free Adaptive Auto Attack (A$^3$) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. Specifically, by observing that adversarial examples to a specific defense model follow some regularities in their starting points, we design an Adaptive Direction Initialization strategy to speed up the evaluation. Furthermore, to approach the lower bound of robustness under the budget number of iterations, we propose an online statistics-based discarding strategy that automatically identifies and abandons hard-to-attack images. Extensive experiments demonstrate the effectiveness of our A$^3$. Particularly, we apply A$^3$ to nearly 50 widely-used defense models. By consuming much fewer iterations than existing methods, i.e., $1/10$ on average (10$\times$ speed up), we achieve lower robust accuracy in all cases. Notably, we won $\textbf{first place}$ out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method. Code is available at: $\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$



## **5. Essential Features: Content-Adaptive Pixel Discretization to Improve Model Robustness to Adaptive Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2012.01699v3)

**Authors**: Ryan Feng, Wu-chi Feng, Atul Prakash

**Abstracts**: Preprocessing defenses such as pixel discretization are appealing to remove adversarial attacks due to their simplicity. However, they have been shown to be ineffective except on simple datasets such as MNIST. We hypothesize that existing discretization approaches failed because using a fixed codebook for the entire dataset limits their ability to balance image representation and codeword separability. We propose a per-image adaptive preprocessing defense called Essential Features, which first applies adaptive blurring to push perturbed pixel values back to their original value and then discretizes the image to an image-adaptive codebook to reduce the color space. Essential Features thus constrains the attack space by forcing the adversary to perturb large regions both locally and color-wise for its effects to survive the preprocessing. Against adaptive attacks, we find that our approach increases the $L_2$ and $L_\infty$ robustness on higher resolution datasets.



## **6. Adversarial Representation Sharing: A Quantitative and Secure Collaborative Learning Framework**

cs.CR

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14299v1)

**Authors**: Jikun Chen, Feng Qiang, Na Ruan

**Abstracts**: The performance of deep learning models highly depends on the amount of training data. It is common practice for today's data holders to merge their datasets and train models collaboratively, which yet poses a threat to data privacy. Different from existing methods such as secure multi-party computation (MPC) and federated learning (FL), we find representation learning has unique advantages in collaborative learning due to the lower communication overhead and task-independency. However, data representations face the threat of model inversion attacks. In this article, we formally define the collaborative learning scenario, and quantify data utility and privacy. Then we present ARS, a collaborative learning framework wherein users share representations of data to train models, and add imperceptible adversarial noise to data representations against reconstruction or attribute extraction attacks. By evaluating ARS in different contexts, we demonstrate that our mechanism is effective against model inversion attacks, and achieves a balance between privacy and utility. The ARS framework has wide applicability. First, ARS is valid for various data types, not limited to images. Second, data representations shared by users can be utilized in different tasks. Third, the framework can be easily extended to the vertical data partitioning scenario.



## **7. Rebuild and Ensemble: Exploring Defense Against Text Adversaries**

cs.CL

work in progress

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14207v1)

**Authors**: Linyang Li, Demin Song, Jiehang Zeng, Ruotian Ma, Xipeng Qiu

**Abstracts**: Adversarial attacks can mislead strong neural models; as such, in NLP tasks, substitution-based attacks are difficult to defend. Current defense methods usually assume that the substitution candidates are accessible, which cannot be widely applied against adversarial attacks unless knowing the mechanism of the attacks. In this paper, we propose a \textbf{Rebuild and Ensemble} Framework to defend against adversarial attacks in texts without knowing the candidates. We propose a rebuild mechanism to train a robust model and ensemble the rebuilt texts during inference to achieve good adversarial defense results. Experiments show that our method can improve accuracy under the current strong attack methods.



## **8. HINT: Hierarchical Neuron Concept Explainer**

cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14196v1)

**Authors**: Andong Wang, Wei-Ning Lee, Xiaojuan Qi

**Abstracts**: To interpret deep networks, one main approach is to associate neurons with human-understandable concepts. However, existing methods often ignore the inherent relationships of different concepts (e.g., dog and cat both belong to animals), and thus lose the chance to explain neurons responsible for higher-level concepts (e.g., animal). In this paper, we study hierarchical concepts inspired by the hierarchical cognition process of human beings. To this end, we propose HIerarchical Neuron concepT explainer (HINT) to effectively build bidirectional associations between neurons and hierarchical concepts in a low-cost and scalable manner. HINT enables us to systematically and quantitatively study whether and how the implicit hierarchical relationships of concepts are embedded into neurons, such as identifying collaborative neurons responsible to one concept and multimodal neurons for different concepts, at different semantic levels from concrete concepts (e.g., dog) to more abstract ones (e.g., animal). Finally, we verify the faithfulness of the associations using Weakly Supervised Object Localization, and demonstrate its applicability in various tasks such as discovering saliency regions and explaining adversarial attacks. Code is available on https://github.com/AntonotnaWang/HINT.



## **9. How to Robustify Black-Box ML Models? A Zeroth-Order Optimization Perspective**

cs.LG

Accepted as ICLR'22 Spotlight Paper

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14195v1)

**Authors**: Yimeng Zhang, Yuguang Yao, Jinghan Jia, Jinfeng Yi, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: The lack of adversarial robustness has been recognized as an important issue for state-of-the-art machine learning (ML) models, e.g., deep neural networks (DNNs). Thereby, robustifying ML models against adversarial attacks is now a major focus of research. However, nearly all existing defense methods, particularly for robust training, made the white-box assumption that the defender has the access to the details of an ML model (or its surrogate alternatives if available), e.g., its architectures and parameters. Beyond existing works, in this paper we aim to address the problem of black-box defense: How to robustify a black-box model using just input queries and output feedback? Such a problem arises in practical scenarios, where the owner of the predictive model is reluctant to share model information in order to preserve privacy. To this end, we propose a general notion of defensive operation that can be applied to black-box models, and design it through the lens of denoised smoothing (DS), a first-order (FO) certified defense technique. To allow the design of merely using model queries, we further integrate DS with the zeroth-order (gradient-free) optimization. However, a direct implementation of zeroth-order (ZO) optimization suffers a high variance of gradient estimates, and thus leads to ineffective defense. To tackle this problem, we next propose to prepend an autoencoder (AE) to a given (black-box) model so that DS can be trained using variance-reduced ZO optimization. We term the eventual defense as ZO-AE-DS. In practice, we empirically show that ZO-AE- DS can achieve improved accuracy, certified robustness, and query complexity over existing baselines. And the effectiveness of our approach is justified under both image classification and image reconstruction tasks. Codes are available at https://github.com/damon-demon/Black-Box-Defense.



## **10. Reverse Engineering of Imperceptible Adversarial Image Perturbations**

cs.CV

**SubmitDate**: 2022-03-26    [paper-pdf](http://arxiv.org/pdf/2203.14145v1)

**Authors**: Yifan Gong, Yuguang Yao, Yize Li, Yimeng Zhang, Xiaoming Liu, Xue Lin, Sijia Liu

**Abstracts**: It has been well recognized that neural network based image classifiers are easily fooled by images with tiny perturbations crafted by an adversary. There has been a vast volume of research to generate and defend such adversarial attacks. However, the following problem is left unexplored: How to reverse-engineer adversarial perturbations from an adversarial image? This leads to a new adversarial learning paradigm--Reverse Engineering of Deceptions (RED). If successful, RED allows us to estimate adversarial perturbations and recover the original images. However, carefully crafted, tiny adversarial perturbations are difficult to recover by optimizing a unilateral RED objective. For example, the pure image denoising method may overfit to minimizing the reconstruction error but hardly preserve the classification properties of the true adversarial perturbations. To tackle this challenge, we formalize the RED problem and identify a set of principles crucial to the RED approach design. Particularly, we find that prediction alignment and proper data augmentation (in terms of spatial transformations) are two criteria to achieve a generalizable RED approach. By integrating these RED principles with image denoising, we propose a new Class-Discriminative Denoising based RED framework, termed CDD-RED. Extensive experiments demonstrate the effectiveness of CDD-RED under different evaluation metrics (ranging from the pixel-level, prediction-level to the attribution-level alignment) and a variety of attack generation methods (e.g., FGSM, PGD, CW, AutoAttack, and adaptive attacks).



## **11. PiDAn: A Coherence Optimization Approach for Backdoor Attack Detection and Mitigation in Deep Neural Networks**

cs.LG

**SubmitDate**: 2022-03-26    [paper-pdf](http://arxiv.org/pdf/2203.09289v2)

**Authors**: Yue Wang, Wenqing Li, Esha Sarkar, Muhammad Shafique, Michail Maniatakos, Saif Eddin Jabari

**Abstracts**: Backdoor attacks impose a new threat in Deep Neural Networks (DNNs), where a backdoor is inserted into the neural network by poisoning the training dataset, misclassifying inputs that contain the adversary trigger. The major challenge for defending against these attacks is that only the attacker knows the secret trigger and the target class. The problem is further exacerbated by the recent introduction of "Hidden Triggers", where the triggers are carefully fused into the input, bypassing detection by human inspection and causing backdoor identification through anomaly detection to fail. To defend against such imperceptible attacks, in this work we systematically analyze how representations, i.e., the set of neuron activations for a given DNN when using the training data as inputs, are affected by backdoor attacks. We propose PiDAn, an algorithm based on coherence optimization purifying the poisoned data. Our analysis shows that representations of poisoned data and authentic data in the target class are still embedded in different linear subspaces, which implies that they show different coherence with some latent spaces. Based on this observation, the proposed PiDAn algorithm learns a sample-wise weight vector to maximize the projected coherence of weighted samples, where we demonstrate that the learned weight vector has a natural "grouping effect" and is distinguishable between authentic data and poisoned data. This enables the systematic detection and mitigation of backdoor attacks. Based on our theoretical analysis and experimental results, we demonstrate the effectiveness of PiDAn in defending against backdoor attacks that use different settings of poisoned samples on GTSRB and ILSVRC2012 datasets. Our PiDAn algorithm can detect more than 90% infected classes and identify 95% poisoned samples.



## **12. A Survey of Robust Adversarial Training in Pattern Recognition: Fundamental, Theory, and Methodologies**

cs.CV

**SubmitDate**: 2022-03-26    [paper-pdf](http://arxiv.org/pdf/2203.14046v1)

**Authors**: Zhuang Qian, Kaizhu Huang, Qiu-Feng Wang, Xu-Yao Zhang

**Abstracts**: In the last a few decades, deep neural networks have achieved remarkable success in machine learning, computer vision, and pattern recognition. Recent studies however show that neural networks (both shallow and deep) may be easily fooled by certain imperceptibly perturbed input samples called adversarial examples. Such security vulnerability has resulted in a large body of research in recent years because real-world threats could be introduced due to vast applications of neural networks. To address the robustness issue to adversarial examples particularly in pattern recognition, robust adversarial training has become one mainstream. Various ideas, methods, and applications have boomed in the field. Yet, a deep understanding of adversarial training including characteristics, interpretations, theories, and connections among different models has still remained elusive. In this paper, we present a comprehensive survey trying to offer a systematic and structured investigation on robust adversarial training in pattern recognition. We start with fundamentals including definition, notations, and properties of adversarial examples. We then introduce a unified theoretical framework for defending against adversarial samples - robust adversarial training with visualizations and interpretations on why adversarial training can lead to model robustness. Connections will be also established between adversarial training and other traditional learning theories. After that, we summarize, review, and discuss various methodologies with adversarial attack and defense/training algorithms in a structured way. Finally, we present analysis, outlook, and remarks of adversarial training.



## **13. Canary Extraction in Natural Language Understanding Models**

cs.CL

Accepted to ACL 2022, Main Conference

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13920v1)

**Authors**: Rahil Parikh, Christophe Dupuy, Rahul Gupta

**Abstracts**: Natural Language Understanding (NLU) models can be trained on sensitive information such as phone numbers, zip-codes etc. Recent literature has focused on Model Inversion Attacks (ModIvA) that can extract training data from model parameters. In this work, we present a version of such an attack by extracting canaries inserted in NLU training data. In the attack, an adversary with open-box access to the model reconstructs the canaries contained in the model's training set. We evaluate our approach by performing text completion on canaries and demonstrate that by using the prefix (non-sensitive) tokens of the canary, we can generate the full canary. As an example, our attack is able to reconstruct a four digit code in the training dataset of the NLU model with a probability of 0.5 in its best configuration. As countermeasures, we identify several defense mechanisms that, when combined, effectively eliminate the risk of ModIvA in our experiments.



## **14. Improving robustness of jet tagging algorithms with adversarial training**

physics.data-an

14 pages, 11 figures, 2 tables

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13890v1)

**Authors**: Annika Stein, Xavier Coubez, Spandan Mondal, Andrzej Novak, Alexander Schmidt

**Abstracts**: Deep learning is a standard tool in the field of high-energy physics, facilitating considerable sensitivity enhancements for numerous analysis strategies. In particular, in identification of physics objects, such as jet flavor tagging, complex neural network architectures play a major role. However, these methods are reliant on accurate simulations. Mismodeling can lead to non-negligible differences in performance in data that need to be measured and calibrated against. We investigate the classifier response to input data with injected mismodelings and probe the vulnerability of flavor tagging algorithms via application of adversarial attacks. Subsequently, we present an adversarial training strategy that mitigates the impact of such simulated attacks and improves the classifier robustness. We examine the relationship between performance and vulnerability and show that this method constitutes a promising approach to reduce the vulnerability to poor modeling.



## **15. Origins of Low-dimensional Adversarial Perturbations**

stat.ML

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13779v1)

**Authors**: Elvis Dohmatob, Chuan Guo, Morgane Goibert

**Abstracts**: In this note, we initiate a rigorous study of the phenomenon of low-dimensional adversarial perturbations in classification. These are adversarial perturbations wherein, unlike the classical setting, the attacker's search is limited to a low-dimensional subspace of the feature space. The goal is to fool the classifier into flipping its decision on a nonzero fraction of inputs from a designated class, upon the addition of perturbations from a subspace chosen by the attacker and fixed once and for all. It is desirable that the dimension $k$ of the subspace be much smaller than the dimension $d$ of the feature space, while the norm of the perturbations should be negligible compared to the norm of a typical data point. In this work, we consider binary classification models under very general regularity conditions, which are verified by certain feedforward neural networks (e.g., with sufficiently smooth, or else ReLU activation function), and compute analytical lower-bounds for the fooling rate of any subspace. These bounds explicitly highlight the dependence that the fooling rate has on the margin of the model (i.e., the ratio of the output to its $L_2$-norm of its gradient at a test point), and on the alignment of the given subspace with the gradients of the model w.r.t. inputs. Our results provide a theoretical explanation for the recent success of heuristic methods for efficiently generating low-dimensional adversarial perturbations. Moreover, our theoretical results are confirmed by experiments.



## **16. Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training**

cs.LG

IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR  2022. Preliminary version ICML 2021 Adversarial Machine Learning Workshop.  Code: https://github.com/TheoT1/FW-AT-Adapt

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2012.12368v5)

**Authors**: Theodoros Tsiligkaridis, Jay Roberts

**Abstracts**: Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense. Due to the high computation time for generating strong adversarial examples in the AT process, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training, and although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical framework for adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the $\ell_2$ distortion of $\ell_\infty$ FW attacks. We analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that $\ell_\infty$ attacks against robust models achieve near maximal distortion, while standard networks have lower distortion. It is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. This mathematical transparency differentiates FW from Projected Gradient Descent (PGD) optimization. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps during training to increase efficiency without compromising robustness. FW-AT-Adapt provides training time on par with single-step fast AT methods and closes the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.



## **17. Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness**

cs.CV

to be published in IEEE/CVF Conference on Computer Vision and Pattern  Recognition 2022, CVPR22

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13639v1)

**Authors**: Giulio Lovisotto, Nicole Finnie, Mauricio Munoz, Chaithanya Kumar Mummadi, Jan Hendrik Metzen

**Abstracts**: Neural architectures based on attention such as vision transformers are revolutionizing image recognition. Their main benefit is that attention allows reasoning about all parts of a scene jointly. In this paper, we show how the global reasoning of (scaled) dot-product attention can be the source of a major vulnerability when confronted with adversarial patch attacks. We provide a theoretical understanding of this vulnerability and relate it to an adversary's ability to misdirect the attention of all queries to a single key token under the control of the adversarial patch. We propose novel adversarial objectives for crafting adversarial patches which target this vulnerability explicitly. We show the effectiveness of the proposed patch attacks on popular image classification (ViTs and DeiTs) and object detection models (DETR). We find that adversarial patches occupying 0.5% of the input can lead to robust accuracies as low as 0% for ViT on ImageNet, and reduce the mAP of DETR on MS COCO to less than 3%.



## **18. Adversarial Bone Length Attack on Action Recognition**

cs.CV

12 pages, 8 figures, accepted to AAAI2022

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2109.05830v2)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: Skeleton-based action recognition models have recently been shown to be vulnerable to adversarial attacks. Compared to adversarial attacks on images, perturbations to skeletons are typically bounded to a lower dimension of approximately 100 per frame. This lower-dimensional setting makes it more difficult to generate imperceptible perturbations. Existing attacks resolve this by exploiting the temporal structure of the skeleton motion so that the perturbation dimension increases to thousands. In this paper, we show that adversarial attacks can be performed on skeleton-based action recognition models, even in a significantly low-dimensional setting without any temporal manipulation. Specifically, we restrict the perturbations to the lengths of the skeleton's bones, which allows an adversary to manipulate only approximately 30 effective dimensions. We conducted experiments on the NTU RGB+D and HDM05 datasets and demonstrate that the proposed attack successfully deceived models with sometimes greater than 90% success rate by small perturbations. Furthermore, we discovered an interesting phenomenon: in our low-dimensional setting, the adversarial training with the bone length attack shares a similar property with data augmentation, and it not only improves the adversarial robustness but also improves the classification accuracy on the original data. This is an interesting counterexample of the trade-off between adversarial robustness and clean accuracy, which has been widely observed in studies on adversarial training in the high-dimensional regime.



## **19. Improving Adversarial Transferability with Spatial Momentum**

cs.CV

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13479v1)

**Authors**: Guoqiu Wang, Xingxing Wei, Huanqian Yan

**Abstracts**: Deep Neural Networks (DNN) are vulnerable to adversarial examples. Although many adversarial attack methods achieve satisfactory attack success rates under the white-box setting, they usually show poor transferability when attacking other DNN models. Momentum-based attack (MI-FGSM) is one effective method to improve transferability. It integrates the momentum term into the iterative process, which can stabilize the update directions by adding the gradients' temporal correlation for each pixel. We argue that only this temporal momentum is not enough, the gradients from the spatial domain within an image, i.e. gradients from the context pixels centered on the target pixel are also important to the stabilization. For that, in this paper, we propose a novel method named Spatial Momentum Iterative FGSM Attack (SMI-FGSM), which introduces the mechanism of momentum accumulation from temporal domain to spatial domain by considering the context gradient information from different regions within the image. SMI-FGSM is then integrated with MI-FGSM to simultaneously stabilize the gradients' update direction from both the temporal and spatial domain. The final method is called SM$^2$I-FGSM. Extensive experiments are conducted on the ImageNet dataset and results show that SM$^2$I-FGSM indeed further enhances the transferability. It achieves the best transferability success rate for multiple mainstream undefended and defended models, which outperforms the state-of-the-art methods by a large margin.



## **20. Bounding Training Data Reconstruction in Private (Deep) Learning**

cs.LG

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2201.12383v2)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.



## **21. A Perturbation Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow**

cs.CV

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2203.13214v1)

**Authors**: Jenny Schmalfuss, Philipp Scholze, Andrés Bruhn

**Abstracts**: Recent optical flow methods are almost exclusively judged in terms of accuracy, while analyzing their robustness is often neglected. Although adversarial attacks offer a useful tool to perform such an analysis, current attacks on optical flow methods rather focus on real-world attacking scenarios than on a worst case robustness assessment. Hence, in this work, we propose a novel adversarial attack - the Perturbation Constrained Flow Attack (PCFA) - that emphasizes destructivity over applicability as a real-world attack. More precisely, PCFA is a global attack that optimizes adversarial perturbations to shift the predicted flow towards a specified target flow, while keeping the L2 norm of the perturbation below a chosen bound. Our experiments not only demonstrate PCFA's applicability in white- and black-box settings, but also show that it finds stronger adversarial samples for optical flow than previous attacking frameworks. Moreover, based on these strong samples, we provide the first common ranking of optical flow methods in the literature considering both prediction quality and adversarial robustness, indicating that high quality methods are not necessarily robust. Our source code will be publicly available.



## **22. Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity**

cs.CV

CVPR 2022 conference (accepted), 18 pages, 17 figure

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2203.05151v4)

**Authors**: Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, Linlin Shen

**Abstracts**: Current adversarial attack research reveals the vulnerability of learning-based classifiers against carefully crafted perturbations. However, most existing attack methods have inherent limitations in cross-dataset generalization as they rely on a classification layer with a closed set of categories. Furthermore, the perturbations generated by these methods may appear in regions easily perceptible to the human visual system (HVS). To circumvent the former problem, we propose a novel algorithm that attacks semantic similarity on feature representations. In this way, we are able to fool classifiers without limiting attacks to a specific dataset. For imperceptibility, we introduce the low-frequency constraint to limit perturbations within high-frequency components, ensuring perceptual similarity between adversarial examples and originals. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) and three public online platforms indicate that our attack can yield misleading and transferable adversarial examples across architectures and datasets. Additionally, visualization results and quantitative performance (in terms of four different metrics) show that the proposed algorithm generates more imperceptible perturbations than the state-of-the-art methods. Code is made available at.



## **23. Imperceptible Transfer Attack and Defense on 3D Point Cloud Classification**

cs.CV

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2111.10990v2)

**Authors**: Daizong Liu, Wei Hu

**Abstracts**: Although many efforts have been made into attack and defense on the 2D image domain in recent years, few methods explore the vulnerability of 3D models. Existing 3D attackers generally perform point-wise perturbation over point clouds, resulting in deformed structures or outliers, which is easily perceivable by humans. Moreover, their adversarial examples are generated under the white-box setting, which frequently suffers from low success rates when transferred to attack remote black-box models. In this paper, we study 3D point cloud attacks from two new and challenging perspectives by proposing a novel Imperceptible Transfer Attack (ITA): 1) Imperceptibility: we constrain the perturbation direction of each point along its normal vector of the neighborhood surface, leading to generated examples with similar geometric properties and thus enhancing the imperceptibility. 2) Transferability: we develop an adversarial transformation model to generate the most harmful distortions and enforce the adversarial examples to resist it, improving their transferability to unknown black-box models. Further, we propose to train more robust black-box 3D models to defend against such ITA attacks by learning more discriminative point cloud representations. Extensive evaluations demonstrate that our ITA attack is more imperceptible and transferable than state-of-the-arts and validate the superiority of our defense strategy.



## **24. Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation**

cs.CL

AAAI 2022

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12709v1)

**Authors**: Hanjie Chen, Yangfeng Ji

**Abstracts**: Neural language models show vulnerability to adversarial examples which are semantically similar to their original counterparts with a few words replaced by their synonyms. A common way to improve model robustness is adversarial training which follows two steps-collecting adversarial examples by attacking a target model, and fine-tuning the model on the augmented dataset with these adversarial examples. The objective of traditional adversarial training is to make a model produce the same correct predictions on an original/adversarial example pair. However, the consistency between model decision-makings on two similar texts is ignored. We argue that a robust model should behave consistently on original/adversarial example pairs, that is making the same predictions (what) based on the same reasons (how) which can be reflected by consistent interpretations. In this work, we propose a novel feature-level adversarial training method named FLAT. FLAT aims at improving model robustness in terms of both predictions and interpretations. FLAT incorporates variational word masks in neural networks to learn global word importance and play as a bottleneck teaching the model to make predictions based on important words. FLAT explicitly shoots at the vulnerability problem caused by the mismatch between model understandings on the replaced words and their synonyms in original/adversarial example pairs by regularizing the corresponding global word importance scores. Experiments show the effectiveness of FLAT in improving the robustness with respect to both predictions and interpretations of four neural network models (LSTM, CNN, BERT, and DeBERTa) to two adversarial attacks on four text classification tasks. The models trained via FLAT also show better robustness than baseline models on unforeseen adversarial examples across different attacks.



## **25. Enhancing Classifier Conservativeness and Robustness by Polynomiality**

cs.LG

IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12693v1)

**Authors**: Ziqi Wang, Marco Loog

**Abstracts**: We illustrate the detrimental effect, such as overconfident decisions, that exponential behavior can have in methods like classical LDA and logistic regression. We then show how polynomiality can remedy the situation. This, among others, leads purposefully to random-level performance in the tails, away from the bulk of the training data. A directly related, simple, yet important technical novelty we subsequently present is softRmax: a reasoned alternative to the standard softmax function employed in contemporary (deep) neural networks. It is derived through linking the standard softmax to Gaussian class-conditional models, as employed in LDA, and replacing those by a polynomial alternative. We show that two aspects of softRmax, conservativeness and inherent gradient regularization, lead to robustness against adversarial attacks without gradient obfuscation.



## **26. Explainability-Aware One Point Attack for Point Cloud Neural Networks**

cs.CV

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2110.04158v3)

**Authors**: Hanxiao Tan, Helena Kotthaus

**Abstracts**: With the proposition of neural networks for point clouds, deep learning has started to shine in the field of 3D object recognition while researchers have shown an increased interest to investigate the reliability of point cloud networks by adversarial attacks. However, most of the existing studies aim to deceive humans or defense algorithms, while the few that address the operation principles of the models themselves remain flawed in terms of critical point selection. In this work, we propose two adversarial methods: One Point Attack (OPA) and Critical Traversal Attack (CTA), which incorporate the explainability technologies and aim to explore the intrinsic operating principle of point cloud networks and their sensitivity against critical points perturbations. Our results show that popular point cloud networks can be deceived with almost $100\%$ success rate by shifting only one point from the input instance. In addition, we show the interesting impact of different point attribution distributions on the adversarial robustness of point cloud networks. Finally, we discuss how our approaches facilitate the explainability study for point cloud networks. To the best of our knowledge, this is the first point-cloud-based adversarial approach concerning explainability. Our code is available at https://github.com/Explain3D/Exp-One-Point-Atk-PC.



## **27. Adversarial Fine-tuning for Backdoor Defense: Connecting Backdoor Attacks to Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2202.06312v2)

**Authors**: Bingxu Mu, Zhenxing Niu, Le Wang, Xue Wang, Rong Jin, Gang Hua

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to both backdoor attacks as well as adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, in this paper we find an intriguing connection between them: for a model planted with backdoors, we observe that its adversarial examples have similar behaviors as its triggered samples, i.e., both activate the same subset of DNN neurons. It indicates that planting a backdoor into a model will significantly affect the model's adversarial examples. Based on this observations, we design a new Adversarial Fine-Tuning (AFT) algorithm to defend against backdoor attacks. We empirically show that, against 5 state-of-the-art backdoor attacks, our AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples and significantly outperforms existing defense methods.



## **28. Input-specific Attention Subnetworks for Adversarial Detection**

cs.CL

Accepted at Findings of ACL 2022, 14 pages, 6 Tables and 9 Figures

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12298v1)

**Authors**: Emil Biju, Anirudh Sriram, Pratyush Kumar, Mitesh M Khapra

**Abstracts**: Self-attention heads are characteristic of Transformer models and have been well studied for interpretability and pruning. In this work, we demonstrate an altogether different utility of attention heads, namely for adversarial detection. Specifically, we propose a method to construct input-specific attention subnetworks (IAS) from which we extract three features to discriminate between authentic and adversarial inputs. The resultant detector significantly improves (by over 7.5%) the state-of-the-art adversarial detection accuracy for the BERT encoder on 10 NLU datasets with 11 different adversarial attack types. We also demonstrate that our method (a) is more accurate for larger models which are likely to have more spurious correlations and thus vulnerable to adversarial attack, and (b) performs well even with modest training sets of adversarial examples.



## **29. Integrity Fingerprinting of DNN with Double Black-box Design and Verification**

cs.CR

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.10902v2)

**Authors**: Shuo Wang, Sharif Abuadbba, Sidharth Agarwal, Kristen Moore, Surya Nepal, Salil Kanhere

**Abstracts**: Cloud-enabled Machine Learning as a Service (MLaaS) has shown enormous promise to transform how deep learning models are developed and deployed. Nonetheless, there is a potential risk associated with the use of such services since a malicious party can modify them to achieve an adverse result. Therefore, it is imperative for model owners, service providers, and end-users to verify whether the deployed model has not been tampered with or not. Such verification requires public verifiability (i.e., fingerprinting patterns are available to all parties, including adversaries) and black-box access to the deployed model via APIs. Existing watermarking and fingerprinting approaches, however, require white-box knowledge (such as gradient) to design the fingerprinting and only support private verifiability, i.e., verification by an honest party.   In this paper, we describe a practical watermarking technique that enables black-box knowledge in fingerprint design and black-box queries during verification. The service ensures the integrity of cloud-based services through public verification (i.e. fingerprinting patterns are available to all parties, including adversaries). If an adversary manipulates a model, this will result in a shift in the decision boundary. Thus, the underlying principle of double-black watermarking is that a model's decision boundary could serve as an inherent fingerprint for watermarking. Our approach captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model's decision boundary in order to capture the inherent fingerprints of the model. We evaluated our watermarking approach against a variety of model integrity attacks and model compression attacks.



## **30. Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**

cs.CV

This paper has been accepted by CVPR2022. Code:  https://github.com/hncszyq/ShadowAttack

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.03818v3)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstracts**: Estimating the risk level of adversarial examples is essential for safely deploying machine learning models in the real world. One popular approach for physical-world attacks is to adopt the "sticker-pasting" strategy, which however suffers from some limitations, including difficulties in access to the target or printing by valid colors. A new type of non-invasive attacks emerged recently, which attempt to cast perturbation onto the target by optics based tools, such as laser beam and projector. However, the added optical patterns are artificial but not natural. Thus, they are still conspicuous and attention-grabbed, and can be easily noticed by humans. In this paper, we study a new type of optical adversarial examples, in which the perturbations are generated by a very common natural phenomenon, shadow, to achieve naturalistic and stealthy physical-world adversarial attack under the black-box setting. We extensively evaluate the effectiveness of this new attack on both simulated and real-world environments. Experimental results on traffic sign recognition demonstrate that our algorithm can generate adversarial examples effectively, reaching 98.23% and 90.47% success rates on LISA and GTSRB test sets respectively, while continuously misleading a moving camera over 95% of the time in real-world scenarios. We also offer discussions about the limitations and the defense mechanism of this attack.



## **31. Online Adversarial Attacks**

cs.LG

ICLR 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2103.02014v4)

**Authors**: Andjela Mladenovic, Avishek Joey Bose, Hugo Berard, William L. Hamilton, Simon Lacoste-Julien, Pascal Vincent, Gauthier Gidel

**Abstracts**: Adversarial attacks expose important vulnerabilities of deep learning models, yet little attention has been paid to settings where data arrives as a stream. In this paper, we formalize the online adversarial attack problem, emphasizing two key elements found in real-world use-cases: attackers must operate under partial knowledge of the target model, and the decisions made by the attacker are irrevocable since they operate on a transient data stream. We first rigorously analyze a deterministic variant of the online threat model by drawing parallels to the well-studied $k$-secretary problem in theoretical computer science and propose Virtual+, a simple yet practical online algorithm. Our main theoretical result shows Virtual+ yields provably the best competitive ratio over all single-threshold algorithms for $k<5$ -- extending the previous analysis of the $k$-secretary problem. We also introduce the \textit{stochastic $k$-secretary} -- effectively reducing online blackbox transfer attacks to a $k$-secretary problem under noise -- and prove theoretical bounds on the performance of Virtual+ adapted to this setting. Finally, we complement our theoretical results by conducting experiments on MNIST, CIFAR-10, and Imagenet classifiers, revealing the necessity of online algorithms in achieving near-optimal performance and also the rich interplay between attack strategies and online attack selection, enabling simple strategies like FGSM to outperform stronger adversaries.



## **32. NNReArch: A Tensor Program Scheduling Framework Against Neural Network Architecture Reverse Engineering**

cs.CR

Accepted by FCCM 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.12046v1)

**Authors**: Yukui Luo, Shijin Duan, Cheng Gongye, Yunsi Fei, Xiaolin Xu

**Abstracts**: Architecture reverse engineering has become an emerging attack against deep neural network (DNN) implementations. Several prior works have utilized side-channel leakage to recover the model architecture while the target is executing on a hardware acceleration platform. In this work, we target an open-source deep-learning accelerator, Versatile Tensor Accelerator (VTA), and utilize electromagnetic (EM) side-channel leakage to comprehensively learn the association between DNN architecture configurations and EM emanations. We also consider the holistic system -- including the low-level tensor program code of the VTA accelerator on a Xilinx FPGA and explore the effect of such low-level configurations on the EM leakage. Our study demonstrates that both the optimization and configuration of tensor programs will affect the EM side-channel leakage.   Gaining knowledge of the association between the low-level tensor program and the EM emanations, we propose NNReArch, a lightweight tensor program scheduling framework against side-channel-based DNN model architecture reverse engineering. Specifically, NNReArch targets reshaping the EM traces of different DNN operators, through scheduling the tensor program execution of the DNN model so as to confuse the adversary. NNReArch is a comprehensive protection framework supporting two modes, a balanced mode that strikes a balance between the DNN model confidentiality and execution performance, and a secure mode where the most secure setting is chosen. We implement and evaluate the proposed framework on the open-source VTA with state-of-the-art DNN architectures. The experimental results demonstrate that NNReArch can efficiently enhance the model architecture security with a small performance overhead. In addition, the proposed obfuscation technique makes reverse engineering of the DNN architecture significantly harder.



## **33. Shape-invariant 3D Adversarial Point Clouds**

cs.CV

Accepted at CVPR 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.04041v2)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Nenghai Yu

**Abstracts**: Adversary and invisibility are two fundamental but conflict characters of adversarial perturbations. Previous adversarial attacks on 3D point cloud recognition have often been criticized for their noticeable point outliers, since they just involve an "implicit constrain" like global distance loss in the time-consuming optimization to limit the generated noise. While point cloud is a highly structured data format, it is hard to constrain its perturbation with a simple loss or metric properly. In this paper, we propose a novel Point-Cloud Sensitivity Map to boost both the efficiency and imperceptibility of point perturbations. This map reveals the vulnerability of point cloud recognition models when encountering shape-invariant adversarial noises. These noises are designed along the shape surface with an "explicit constrain" instead of extra distance loss. Specifically, we first apply a reversible coordinate transformation on each point of the point cloud input, to reduce one degree of point freedom and limit its movement on the tangent plane. Then we calculate the best attacking direction with the gradients of the transformed point cloud obtained on the white-box model. Finally we assign each point with a non-negative score to construct the sensitivity map, which benefits both white-box adversarial invisibility and black-box query-efficiency extended in our work. Extensive evaluations prove that our method can achieve the superior performance on various point cloud recognition models, with its satisfying adversarial imperceptibility and strong resistance to different point cloud defense settings. Our code is available at: https://github.com/shikiw/SI-Adv.



## **34. Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis**

cs.LG

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11633v1)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstracts**: Model poisoning attacks on federated learning (FL) intrude in the entire system via compromising an edge model, resulting in malfunctioning of machine learning models. Such compromised models are tampered with to perform adversary-desired behaviors. In particular, we considered a semi-targeted situation where the source class is predetermined however the target class is not. The goal is to cause the global classifier to misclassify data of the source class. Though approaches such as label flipping have been adopted to inject poisoned parameters into FL, it has been shown that their performances are usually class-sensitive varying with different target classes applied. Typically, an attack can become less effective when shifting to a different target class. To overcome this challenge, we propose the Attacking Distance-aware Attack (ADA) to enhance a poisoning attack by finding the optimized target class in the feature space. Moreover, we studied a more challenging situation where an adversary had limited prior knowledge about a client's data. To tackle this problem, ADA deduces pair-wise distances between different classes in the latent feature space from shared model parameters based on the backward error analysis. We performed extensive empirical evaluations on ADA by varying the factor of attacking frequency in three different image classification tasks. As a result, ADA succeeded in increasing the attack performance by 1.8 times in the most challenging case with an attacking frequency of 0.01.



## **35. Exploring High-Order Structure for Robust Graph Structure Learning**

cs.LG

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11492v1)

**Authors**: Guangqian Yang, Yibing Zhan, Jinlong Li, Baosheng Yu, Liu Liu, Fengxiang He

**Abstracts**: Recent studies show that Graph Neural Networks (GNNs) are vulnerable to adversarial attack, i.e., an imperceptible structure perturbation can fool GNNs to make wrong predictions. Some researches explore specific properties of clean graphs such as the feature smoothness to defense the attack, but the analysis of it has not been well-studied. In this paper, we analyze the adversarial attack on graphs from the perspective of feature smoothness which further contributes to an efficient new adversarial defensive algorithm for GNNs. We discover that the effect of the high-order graph structure is a smoother filter for processing graph structures. Intuitively, the high-order graph structure denotes the path number between nodes, where larger number indicates closer connection, so it naturally contributes to defense the adversarial perturbation. Further, we propose a novel algorithm that incorporates the high-order structural information into the graph structure learning. We perform experiments on three popular benchmark datasets, Cora, Citeseer and Polblogs. Extensive experiments demonstrate the effectiveness of our method for defending against graph adversarial attacks.



## **36. Making DeepFakes more spurious: evading deep face forgery detection via trace removal attack**

cs.CV

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11433v1)

**Authors**: Chi Liu, Huajie Chen, Tianqing Zhu, Jun Zhang, Wanlei Zhou

**Abstracts**: DeepFakes are raising significant social concerns. Although various DeepFake detectors have been developed as forensic countermeasures, these detectors are still vulnerable to attacks. Recently, a few attacks, principally adversarial attacks, have succeeded in cloaking DeepFake images to evade detection. However, these attacks have typical detector-specific designs, which require prior knowledge about the detector, leading to poor transferability. Moreover, these attacks only consider simple security scenarios. Less is known about how effective they are in high-level scenarios where either the detectors or the attacker's knowledge varies. In this paper, we solve the above challenges with presenting a novel detector-agnostic trace removal attack for DeepFake anti-forensics. Instead of investigating the detector side, our attack looks into the original DeepFake creation pipeline, attempting to remove all detectable natural DeepFake traces to render the fake images more "authentic". To implement this attack, first, we perform a DeepFake trace discovery, identifying three discernible traces. Then a trace removal network (TR-Net) is proposed based on an adversarial learning framework involving one generator and multiple discriminators. Each discriminator is responsible for one individual trace representation to avoid cross-trace interference. These discriminators are arranged in parallel, which prompts the generator to remove various traces simultaneously. To evaluate the attack efficacy, we crafted heterogeneous security scenarios where the detectors were embedded with different levels of defense and the attackers' background knowledge of data varies. The experimental results show that the proposed attack can significantly compromise the detection accuracy of six state-of-the-art DeepFake detectors while causing only a negligible loss in visual quality to the original DeepFake samples.



## **37. Subspace Adversarial Training**

cs.LG

CVPR2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2111.12229v2)

**Authors**: Tao Li, Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: Single-step adversarial training (AT) has received wide attention as it proved to be both efficient and robust. However, a serious problem of catastrophic overfitting exists, i.e., the robust accuracy against projected gradient descent (PGD) attack suddenly drops to 0% during the training. In this paper, we approach this problem from a novel perspective of optimization and firstly reveal the close link between the fast-growing gradient of each sample and overfitting, which can also be applied to understand robust overfitting in multi-step AT. To control the growth of the gradient, we propose a new AT method, Subspace Adversarial Training (Sub-AT), which constrains AT in a carefully extracted subspace. It successfully resolves both kinds of overfitting and significantly boosts the robustness. In subspace, we also allow single-step AT with larger steps and larger radius, further improving the robustness performance. As a result, we achieve state-of-the-art single-step AT performance. Without any regularization term, our single-step AT can reach over 51% robust accuracy against strong PGD-50 attack of radius 8/255 on CIFAR-10, reaching a competitive performance against standard multi-step PGD-10 AT with huge computational advantages. The code is released at https://github.com/nblt/Sub-AT.



## **38. On The Robustness of Offensive Language Classifiers**

cs.CL

9 pages, 2 figures, Accepted at ACL 2022

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.11331v1)

**Authors**: Jonathan Rusert, Zubair Shafiq, Padmini Srinivasan

**Abstracts**: Social media platforms are deploying machine learning based offensive language classification systems to combat hateful, racist, and other forms of offensive speech at scale. However, despite their real-world deployment, we do not yet comprehensively understand the extent to which offensive language classifiers are robust against adversarial attacks. Prior work in this space is limited to studying robustness of offensive language classifiers against primitive attacks such as misspellings and extraneous spaces. To address this gap, we systematically analyze the robustness of state-of-the-art offensive language classifiers against more crafty adversarial attacks that leverage greedy- and attention-based word selection and context-aware embeddings for word replacement. Our results on multiple datasets show that these crafty adversarial attacks can degrade the accuracy of offensive language classifiers by more than 50% while also being able to preserve the readability and meaning of the modified text.



## **39. FGAN: Federated Generative Adversarial Networks for Anomaly Detection in Network Traffic**

cs.CR

8 pages, 2 figures

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.11106v1)

**Authors**: Sankha Das

**Abstracts**: Over the last two decades, a lot of work has been done in improving network security, particularly in intrusion detection systems (IDS) and anomaly detection. Machine learning solutions have also been employed in IDSs to detect known and plausible attacks in incoming traffic. Parameters such as packet contents, sender IP and sender port, connection duration, etc. have been previously used to train these machine learning models to learn to differentiate genuine traffic from malicious ones. Generative Adversarial Networks (GANs) have been significantly successful in detecting such anomalies, mostly attributed to the adversarial training of the generator and discriminator in an attempt to bypass each other and in turn increase their own power and accuracy. However, in large networks having a wide variety of traffic at possibly different regions of the network and susceptible to a large number of potential attacks, training these GANs for a particular kind of anomaly may make it oblivious to other anomalies and attacks. In addition, the dataset required to train these models has to be made centrally available and publicly accessible, posing the obvious question of privacy of the communications of the respective participants of the network. The solution proposed in this work aims at tackling the above two issues by using GANs in a federated architecture in networks of such scale and capacity. In such a setting, different users of the network will be able to train and customize a centrally available adversarial model according to their own frequently faced conditions. Simultaneously, the member users of the network will also able to gain from the experiences of the other users in the network.



## **40. An Intermediate-level Attack Framework on The Basis of Linear Regression**

cs.CV

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10723v1)

**Authors**: Yiwen Guo, Qizhang Li, Wangmeng Zuo, Hao Chen

**Abstracts**: This paper substantially extends our work published at ECCV, in which an intermediate-level attack was proposed to improve the transferability of some baseline adversarial examples. We advocate to establish a direct linear mapping from the intermediate-level discrepancies (between adversarial features and benign features) to classification prediction loss of the adversarial example. In this paper, we delve deep into the core components of such a framework by performing comprehensive studies and extensive experiments. We show that 1) a variety of linear regression models can all be considered in order to establish the mapping, 2) the magnitude of the finally obtained intermediate-level discrepancy is linearly correlated with adversarial transferability, 3) further boost of the performance can be achieved by performing multiple runs of the baseline attack with random initialization. By leveraging these findings, we achieve new state-of-the-arts on transfer-based $\ell_\infty$ and $\ell_2$ attacks.



## **41. A Prompting-based Approach for Adversarial Example Generation and Robustness Enhancement**

cs.CL

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10714v1)

**Authors**: Yuting Yang, Pei Huang, Juan Cao, Jintao Li, Yun Lin, Jin Song Dong, Feifei Ma, Jian Zhang

**Abstracts**: Recent years have seen the wide application of NLP models in crucial areas such as finance, medical treatment, and news media, raising concerns of the model robustness and vulnerabilities. In this paper, we propose a novel prompt-based adversarial attack to compromise NLP models and robustness enhancement technique. We first construct malicious prompts for each instance and generate adversarial examples via mask-and-filling under the effect of a malicious purpose. Our attack technique targets the inherent vulnerabilities of NLP models, allowing us to generate samples even without interacting with the victim NLP model, as long as it is based on pre-trained language models (PLMs). Furthermore, we design a prompt-based adversarial training method to improve the robustness of PLMs. As our training method does not actually generate adversarial samples, it can be applied to large-scale training sets efficiently. The experimental results show that our attack method can achieve a high attack success rate with more diverse, fluent and natural adversarial examples. In addition, our robustness enhancement method can significantly improve the robustness of models to resist adversarial attacks. Our work indicates that prompting paradigm has great potential in probing some fundamental flaws of PLMs and fine-tuning them for downstream tasks.



## **42. Leveraging Expert Guided Adversarial Augmentation For Improving Generalization in Named Entity Recognition**

cs.CL

ACL 2022 (Findings)

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10693v1)

**Authors**: Aaron Reich, Jiaao Chen, Aastha Agrawal, Yanzhe Zhang, Diyi Yang

**Abstracts**: Named Entity Recognition (NER) systems often demonstrate great performance on in-distribution data, but perform poorly on examples drawn from a shifted distribution. One way to evaluate the generalization ability of NER models is to use adversarial examples, on which the specific variations associated with named entities are rarely considered. To this end, we propose leveraging expert-guided heuristics to change the entity tokens and their surrounding contexts thereby altering their entity types as adversarial attacks. Using expert-guided heuristics, we augmented the CoNLL 2003 test set and manually annotated it to construct a high-quality challenging set. We found that state-of-the-art NER systems trained on CoNLL 2003 training data drop performance dramatically on our challenging set. By training on adversarial augmented training examples and using mixup for regularization, we were able to significantly improve the performance on the challenging set as well as improve out-of-domain generalization which we evaluated by using OntoNotes data. We have publicly released our dataset and code at https://github.com/GT-SALT/Guided-Adversarial-Augmentation.



## **43. RareGAN: Generating Samples for Rare Classes**

cs.LG

Published in AAAI 2022

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10674v1)

**Authors**: Zinan Lin, Hao Liang, Giulia Fanti, Vyas Sekar

**Abstracts**: We study the problem of learning generative adversarial networks (GANs) for a rare class of an unlabeled dataset subject to a labeling budget. This problem is motivated from practical applications in domains including security (e.g., synthesizing packets for DNS amplification attacks), systems and networking (e.g., synthesizing workloads that trigger high resource usage), and machine learning (e.g., generating images from a rare class). Existing approaches are unsuitable, either requiring fully-labeled datasets or sacrificing the fidelity of the rare class for that of the common classes. We propose RareGAN, a novel synthesis of three key ideas: (1) extending conditional GANs to use labelled and unlabelled data for better generalization; (2) an active learning approach that requests the most useful labels; and (3) a weighted loss function to favor learning the rare class. We show that RareGAN achieves a better fidelity-diversity tradeoff on the rare class than prior work across different applications, budgets, rare class fractions, GAN losses, and architectures.



## **44. Does DQN really learn? Exploring adversarial training schemes in Pong**

cs.LG

RLDM 2022

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10614v1)

**Authors**: Bowen He, Sreehari Rammohan, Jessica Forde, Michael Littman

**Abstracts**: In this work, we study two self-play training schemes, Chainer and Pool, and show they lead to improved agent performance in Atari Pong compared to a standard DQN agent -- trained against the built-in Atari opponent. To measure agent performance, we define a robustness metric that captures how difficult it is to learn a strategy that beats the agent's learned policy. Through playing past versions of themselves, Chainer and Pool are able to target weaknesses in their policies and improve their resistance to attack. Agents trained using these methods score well on our robustness metric and can easily defeat the standard DQN agent. We conclude by using linear probing to illuminate what internal structures the different agents develop to play the game. We show that training agents with Chainer or Pool leads to richer network activations with greater predictive power to estimate critical game-state features compared to the standard DQN agent.



## **45. Improved Semi-Quantum Key Distribution with Two Almost-Classical Users**

quant-ph

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10567v1)

**Authors**: Saachi Mutreja, Walter O. Krawec

**Abstracts**: Semi-quantum key distribution (SQKD) protocols attempt to establish a shared secret key between users, secure against computationally unbounded adversaries. Unlike standard quantum key distribution protocols, SQKD protocols contain at least one user who is limited in their quantum abilities and is almost "classical" in nature. In this paper, we revisit a mediated semi-quantum key distribution protocol, introduced by Massa et al., in 2019, where users need only the ability to detect a qubit, or reflect a qubit; they do not need to perform any other basis measurement; nor do they need to prepare quantum signals. Users require the services of a quantum server which may be controlled by the adversary. In this paper, we show how this protocol may be extended to improve its efficiency and also its noise tolerance. We discuss an extension which allows more communication rounds to be directly usable; we analyze the key-rate of this extension in the asymptotic scenario for a particular class of attacks and compare with prior work. Finally, we evaluate the protocol's performance in a variety of lossy and noisy channels.



## **46. Adversarial Parameter Attack on Deep Neural Networks**

cs.LG

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10502v1)

**Authors**: Lijia Yu, Yihan Wang, Xiao-Shan Gao

**Abstracts**: In this paper, a new parameter perturbation attack on DNNs, called adversarial parameter attack, is proposed, in which small perturbations to the parameters of the DNN are made such that the accuracy of the attacked DNN does not decrease much, but its robustness becomes much lower. The adversarial parameter attack is stronger than previous parameter perturbation attacks in that the attack is more difficult to be recognized by users and the attacked DNN gives a wrong label for any modified sample input with high probability. The existence of adversarial parameters is proved. For a DNN $F_{\Theta}$ with the parameter set $\Theta$ satisfying certain conditions, it is shown that if the depth of the DNN is sufficiently large, then there exists an adversarial parameter set $\Theta_a$ for $\Theta$ such that the accuracy of $F_{\Theta_a}$ is equal to that of $F_{\Theta}$, but the robustness measure of $F_{\Theta_a}$ is smaller than any given bound. An effective training algorithm is given to compute adversarial parameters and numerical experiments are used to demonstrate that the algorithms are effective to produce high quality adversarial parameters.



## **47. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

cs.LG

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.08959v2)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.



## **48. On Robust Prefix-Tuning for Text Classification**

cs.CL

Accepted in ICLR 2022. We release the code at  https://github.com/minicheshire/Robust-Prefix-Tuning

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10378v1)

**Authors**: Zonghan Yang, Yang Liu

**Abstracts**: Recently, prefix-tuning has gained increasing attention as a parameter-efficient finetuning method for large-scale pretrained language models. The method keeps the pretrained models fixed and only updates the prefix token parameters for each downstream task. Despite being lightweight and modular, prefix-tuning still lacks robustness to textual adversarial attacks. However, most currently developed defense techniques necessitate auxiliary model update and storage, which inevitably hamper the modularity and low storage of prefix-tuning. In this work, we propose a robust prefix-tuning framework that preserves the efficiency and modularity of prefix-tuning. The core idea of our framework is leveraging the layerwise activations of the language model by correctly-classified training data as the standard for additional prefix finetuning. During the test phase, an extra batch-level prefix is tuned for each batch and added to the original prefix for robustness enhancement. Extensive experiments on three text classification benchmarks show that our framework substantially improves robustness over several strong baselines against five textual attacks of different types while maintaining comparable accuracy on clean texts. We also interpret our robust prefix-tuning framework from the optimal control perspective and pose several directions for future research.



## **49. Perturbations in the Wild: Leveraging Human-Written Text Perturbations for Realistic Adversarial Attack and Defense**

cs.LG

Accepted to the 60th Annual Meeting of the Association for  Computational Linguistics (ACL'22), Findings

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10346v1)

**Authors**: Thai Le, Jooyoung Lee, Kevin Yen, Yifan Hu, Dongwon Lee

**Abstracts**: We proposes a novel algorithm, ANTHRO, that inductively extracts over 600K human-written text perturbations in the wild and leverages them for realistic adversarial attack. Unlike existing character-based attacks which often deductively hypothesize a set of manipulation strategies, our work is grounded on actual observations from real-world texts. We find that adversarial texts generated by ANTHRO achieve the best trade-off between (1) attack success rate, (2) semantic preservation of the original text, and (3) stealthiness--i.e. indistinguishable from human writings hence harder to be flagged as suspicious. Specifically, our attacks accomplished around 83% and 91% attack success rates on BERT and RoBERTa, respectively. Moreover, it outperformed the TextBugger baseline with an increase of 50% and 40% in terms of semantic preservation and stealthiness when evaluated by both layperson and professional human workers. ANTHRO can further enhance a BERT classifier's performance in understanding different variations of human-written toxic texts via adversarial training when compared to the Perspective API.



## **50. Efficient Neural Network Analysis with Sum-of-Infeasibilities**

cs.LG

TACAS'22

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.11201v1)

**Authors**: Haoze Wu, Aleksandar Zeljić, Guy Katz, Clark Barrett

**Abstracts**: Inspired by sum-of-infeasibilities methods in convex optimization, we propose a novel procedure for analyzing verification queries on neural networks with piecewise-linear activation functions. Given a convex relaxation which over-approximates the non-convex activation functions, we encode the violations of activation functions as a cost function and optimize it with respect to the convex relaxation. The cost function, referred to as the Sum-of-Infeasibilities (SoI), is designed so that its minimum is zero and achieved only if all the activation functions are satisfied. We propose a stochastic procedure, DeepSoI, to efficiently minimize the SoI. An extension to a canonical case-analysis-based complete search procedure can be achieved by replacing the convex procedure executed at each search state with DeepSoI. Extending the complete search with DeepSoI achieves multiple simultaneous goals: 1) it guides the search towards a counter-example; 2) it enables more informed branching decisions; and 3) it creates additional opportunities for bound derivation. An extensive evaluation across different benchmarks and solvers demonstrates the benefit of the proposed techniques. In particular, we demonstrate that SoI significantly improves the performance of an existing complete search procedure. Moreover, the SoI-based implementation outperforms other state-of-the-art complete verifiers. We also show that our technique can efficiently improve upon the perturbation bound derived by a recent adversarial attack algorithm.



