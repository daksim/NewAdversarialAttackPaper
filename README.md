# Latest Adversarial Attack Papers
**update at 2023-03-14 09:29:06**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Symmetry Defense Against CNN Adversarial Perturbation Attacks**

cs.LG

13 pages

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2210.04087v2) [paper-pdf](http://arxiv.org/pdf/2210.04087v2)

**Authors**: Blerta Lindqvist

**Abstract**: Convolutional neural network classifiers (CNNs) are susceptible to adversarial attacks that perturb original samples to fool classifiers such as an autonomous vehicle's road sign image classifier. CNNs also lack invariance in the classification of symmetric samples because CNNs can classify symmetric samples differently. Considered together, the CNN lack of adversarial robustness and the CNN lack of invariance mean that the classification of symmetric adversarial samples can differ from their incorrect classification. Could symmetric adversarial samples revert to their correct classification? This paper answers this question by designing a symmetry defense that inverts or horizontally flips adversarial samples before classification against adversaries unaware of the defense. Against adversaries aware of the defense, the defense devises a Klein four symmetry subgroup that includes the horizontal flip and pixel inversion symmetries. The symmetry defense uses the subgroup symmetries in accuracy evaluation and the subgroup closure property to confine the transformations that an adaptive adversary can apply before or after generating the adversarial sample. Without changing the preprocessing, parameters, or model, the proposed symmetry defense counters the Projected Gradient Descent (PGD) and AutoAttack attacks with near-default accuracies for ImageNet. Without using attack knowledge or adversarial samples, the proposed defense exceeds the current best defense, which trains on adversarial samples. The defense maintains and even improves the classification accuracy of non-adversarial samples.



## **2. Review on the Feasibility of Adversarial Evasion Attacks and Defenses for Network Intrusion Detection Systems**

cs.CR

Under review (Submitted to Computer Networks - Elsevier)

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.07003v1) [paper-pdf](http://arxiv.org/pdf/2303.07003v1)

**Authors**: Islam Debicha, Benjamin Cochez, Tayeb Kenaza, Thibault Debatty, Jean-Michel Dricot, Wim Mees

**Abstract**: Nowadays, numerous applications incorporate machine learning (ML) algorithms due to their prominent achievements. However, many studies in the field of computer vision have shown that ML can be fooled by intentionally crafted instances, called adversarial examples. These adversarial examples take advantage of the intrinsic vulnerability of ML models. Recent research raises many concerns in the cybersecurity field. An increasing number of researchers are studying the feasibility of such attacks on security systems based on ML algorithms, such as Intrusion Detection Systems (IDS). The feasibility of such adversarial attacks would be influenced by various domain-specific constraints. This can potentially increase the difficulty of crafting adversarial examples. Despite the considerable amount of research that has been done in this area, much of it focuses on showing that it is possible to fool a model using features extracted from the raw data but does not address the practical side, i.e., the reverse transformation from theory to practice. For this reason, we propose a review browsing through various important papers to provide a comprehensive analysis. Our analysis highlights some challenges that have not been addressed in the reviewed papers.



## **3. Towards Making a Trojan-horse Attack on Text-to-Image Retrieval**

cs.MM

Accepted by ICASSP 2023

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2202.03861v4) [paper-pdf](http://arxiv.org/pdf/2202.03861v4)

**Authors**: Fan Hu, Aozhu Chen, Xirong Li

**Abstract**: While deep learning based image retrieval is reported to be vulnerable to adversarial attacks, existing works are mainly on image-to-image retrieval with their attacks performed at the front end via query modification. By contrast, we present in this paper the first study about a threat that occurs at the back end of a text-to-image retrieval (T2IR) system. Our study is motivated by the fact that the image collection indexed by the system will be regularly updated due to the arrival of new images from various sources such as web crawlers and advertisers. With malicious images indexed, it is possible for an attacker to indirectly interfere with the retrieval process, letting users see certain images that are completely irrelevant w.r.t. their queries. We put this thought into practice by proposing a novel Trojan-horse attack (THA). In particular, we construct a set of Trojan-horse images by first embedding word-specific adversarial information into a QR code and then putting the code on benign advertising images. A proof-of-concept evaluation, conducted on two popular T2IR datasets (Flickr30k and MS-COCO), shows the effectiveness of the proposed THA in a white-box mode.



## **4. Robust Contrastive Language-Image Pretraining against Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.06854v1) [paper-pdf](http://arxiv.org/pdf/2303.06854v1)

**Authors**: Wenhan Yang, Baharan Mirzasoleiman

**Abstract**: Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of adversarial attacks, including targeted and backdoor data poisoning attacks. Despite this vulnerability, robust contrastive vision-language pretraining against adversarial attacks has remained unaddressed. In this work, we propose RoCLIP, the first effective method for robust pretraining {and fine-tuning} multimodal vision-language models. RoCLIP effectively breaks the association between poisoned image-caption pairs by considering a pool of random examples, and (1) matching every image with the text that is most similar to its caption in the pool, and (2) matching every caption with the image that is most similar to its image in the pool. Our extensive experiments show that our method renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training or fine-tuning of CLIP. In particular, RoCLIP decreases the poison and backdoor attack success rates down to 0\% during pre-training and 1\%-4\% during fine-tuning, and effectively improves the model's performance.



## **5. Adversarial Attacks to Direct Data-driven Control for Destabilization**

eess.SY

6 pages

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.06837v1) [paper-pdf](http://arxiv.org/pdf/2303.06837v1)

**Authors**: Hampei Sasahara

**Abstract**: This study investigates the vulnerability of direct data-driven control to adversarial attacks in the form of a small but sophisticated perturbation added to the original data. The directed gradient sign method (DGSM) is developed as a specific attack method, based on the fast gradient sign method (FGSM), which has originally been considered in image classification. DGSM uses the gradient of the eigenvalues of the resulting closed-loop system and crafts a perturbation in the direction where the system becomes less stable. It is demonstrated that the system can be destabilized by the attack, even if the original closed-loop system with the clean data has a large margin of stability. To increase the robustness against the attack, regularization methods that have been developed to deal with random disturbances are considered. Their effectiveness is evaluated by numerical experiments using an inverted pendulum model.



## **6. Protecting Quantum Procrastinators with Signature Lifting: A Case Study in Cryptocurrencies**

cs.CR

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06754v1) [paper-pdf](http://arxiv.org/pdf/2303.06754v1)

**Authors**: Or Sattath, Shai Wyborski

**Abstract**: Current solutions to quantum vulnerabilities of widely used cryptographic schemes involve migrating users to post-quantum schemes before quantum attacks become feasible. This work deals with protecting quantum procrastinators: users that failed to migrate to post-quantum cryptography in time.   To address this problem in the context of digital signatures, we introduce a technique called signature lifting, that allows us to lift a deployed pre-quantum signature scheme satisfying a certain property to a post-quantum signature scheme that uses the same keys. Informally, the said property is that a post-quantum one-way function is used "somewhere along the way" to derive the public-key from the secret-key. Our constructions of signature lifting relies heavily on the post-quantum digital signature scheme Picnic (Chase et al., CCS'17).   Our main case-study is cryptocurrencies, where this property holds in two scenarios: when the public-key is generated via a key-derivation function or when the public-key hash is posted instead of the public-key itself. We propose a modification, based on signature lifting, that can be applied in many cryptocurrencies for securely spending pre-quantum coins in presence of quantum adversaries. Our construction improves upon existing constructions in two major ways: it is not limited to pre-quantum coins whose ECDSA public-key has been kept secret (and in particular, it handles all coins that are stored in addresses generated by HD wallets), and it does not require access to post-quantum coins or using side payments to pay for posting the transaction.



## **7. DNN-Alias: Deep Neural Network Protection Against Side-Channel Attacks via Layer Balancing**

cs.CR

10 pages

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06746v1) [paper-pdf](http://arxiv.org/pdf/2303.06746v1)

**Authors**: Mahya Morid Ahmadi, Lilas Alrahis, Ozgur Sinanoglu, Muhammad Shafique

**Abstract**: Extracting the architecture of layers of a given deep neural network (DNN) through hardware-based side channels allows adversaries to steal its intellectual property and even launch powerful adversarial attacks on the target system. In this work, we propose DNN-Alias, an obfuscation method for DNNs that forces all the layers in a given network to have similar execution traces, preventing attack models from differentiating between the layers. Towards this, DNN-Alias performs various layer-obfuscation operations, e.g., layer branching, layer deepening, etc, to alter the run-time traces while maintaining the functionality. DNN-Alias deploys an evolutionary algorithm to find the best combination of obfuscation operations in terms of maximizing the security level while maintaining a user-provided latency overhead budget. We demonstrate the effectiveness of our DNN-Alias technique by obfuscating the architecture of 700 randomly generated and obfuscated DNNs running on multiple Nvidia RTX 2080 TI GPU-based machines. Our experiments show that state-of-the-art side-channel architecture stealing attacks cannot extract the original DNN accurately. Moreover, we obfuscate the architecture of various DNNs, such as the VGG-11, VGG-13, ResNet-20, and ResNet-32 networks. Training the DNNs using the standard CIFAR10 dataset, we show that our DNN-Alias maintains the functionality of the original DNNs by preserving the original inference accuracy. Further, the experiments highlight that adversarial attack on obfuscated DNNs is unsuccessful.



## **8. Adv-Bot: Realistic Adversarial Botnet Attacks against Network Intrusion Detection Systems**

cs.CR

This work is published in Computers & Security (an Elsevier journal)  https://www.sciencedirect.com/science/article/pii/S016740482300086X

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06664v1) [paper-pdf](http://arxiv.org/pdf/2303.06664v1)

**Authors**: Islam Debicha, Benjamin Cochez, Tayeb Kenaza, Thibault Debatty, Jean-Michel Dricot, Wim Mees

**Abstract**: Due to the numerous advantages of machine learning (ML) algorithms, many applications now incorporate them. However, many studies in the field of image classification have shown that MLs can be fooled by a variety of adversarial attacks. These attacks take advantage of ML algorithms' inherent vulnerability. This raises many questions in the cybersecurity field, where a growing number of researchers are recently investigating the feasibility of such attacks against machine learning-based security systems, such as intrusion detection systems. The majority of this research demonstrates that it is possible to fool a model using features extracted from a raw data source, but it does not take into account the real implementation of such attacks, i.e., the reverse transformation from theory to practice. The real implementation of these adversarial attacks would be influenced by various constraints that would make their execution more difficult. As a result, the purpose of this study was to investigate the actual feasibility of adversarial attacks, specifically evasion attacks, against network-based intrusion detection systems (NIDS), demonstrating that it is entirely possible to fool these ML-based IDSs using our proposed adversarial algorithm while assuming as many constraints as possible in a black-box setting. In addition, since it is critical to design defense mechanisms to protect ML-based IDSs against such attacks, a defensive scheme is presented. Realistic botnet traffic traces are used to assess this work. Our goal is to create adversarial botnet traffic that can avoid detection while still performing all of its intended malicious functionality.



## **9. Interpreting Hidden Semantics in the Intermediate Layers of 3D Point Cloud Classification Neural Network**

cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06652v1) [paper-pdf](http://arxiv.org/pdf/2303.06652v1)

**Authors**: Weiquan Liu, Minghao Liu, Shijun Zheng, Cheng Wang

**Abstract**: Although 3D point cloud classification neural network models have been widely used, the in-depth interpretation of the activation of the neurons and layers is still a challenge. We propose a novel approach, named Relevance Flow, to interpret the hidden semantics of 3D point cloud classification neural networks. It delivers the class Relevance to the activated neurons in the intermediate layers in a back-propagation manner, and associates the activation of neurons with the input points to visualize the hidden semantics of each layer. Specially, we reveal that the 3D point cloud classification neural network has learned the plane-level and part-level hidden semantics in the intermediate layers, and utilize the normal and IoU to evaluate the consistency of both levels' hidden semantics. Besides, by using the hidden semantics, we generate the adversarial attack samples to attack 3D point cloud classifiers. Experiments show that our proposed method reveals the hidden semantics of the 3D point cloud classification neural network on ModelNet40 and ShapeNet, which can be used for the unsupervised point cloud part segmentation without labels and attacking the 3D point cloud classifiers.



## **10. Query Attack by Multi-Identity Surrogates**

cs.LG

IEEE TRANSACTIONS ON ARTIFICIAL INTELLIGENCE

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2105.15010v5) [paper-pdf](http://arxiv.org/pdf/2105.15010v5)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Xiaolin Huang

**Abstract**: Deep Neural Networks (DNNs) are acknowledged as vulnerable to adversarial attacks, while the existing black-box attacks require extensive queries on the victim DNN to achieve high success rates. For query-efficiency, surrogate models of the victim are used to generate transferable Adversarial Examples (AEs) because of their Gradient Similarity (GS), i.e., surrogates' attack gradients are similar to the victim's ones. However, it is generally neglected to exploit their similarity on outputs, namely the Prediction Similarity (PS), to filter out inefficient queries by surrogates without querying the victim. To jointly utilize and also optimize surrogates' GS and PS, we develop QueryNet, a unified attack framework that can significantly reduce queries. QueryNet creatively attacks by multi-identity surrogates, i.e., crafts several AEs for one sample by different surrogates, and also uses surrogates to decide on the most promising AE for the query. After that, the victim's query feedback is accumulated to optimize not only surrogates' parameters but also their architectures, enhancing both the GS and the PS. Although QueryNet has no access to pre-trained surrogates' prior, it reduces queries by averagely about an order of magnitude compared to alternatives within an acceptable time, according to our comprehensive experiments: 11 victims (including two commercial models) on MNIST/CIFAR10/ImageNet, allowing only 8-bit image queries, and no access to the victim's training data. The code is available at https://github.com/Sizhe-Chen/QueryNet.



## **11. Adaptive Local Adversarial Attacks on 3D Point Clouds for Augmented Reality**

cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06641v1) [paper-pdf](http://arxiv.org/pdf/2303.06641v1)

**Authors**: Weiquan Liu, Shijun Zheng, Cheng Wang

**Abstract**: As the key technology of augmented reality (AR), 3D recognition and tracking are always vulnerable to adversarial examples, which will cause serious security risks to AR systems. Adversarial examples are beneficial to improve the robustness of the 3D neural network model and enhance the stability of the AR system. At present, most 3D adversarial attack methods perturb the entire point cloud to generate adversarial examples, which results in high perturbation costs and difficulty in reconstructing the corresponding real objects in the physical world. In this paper, we propose an adaptive local adversarial attack method (AL-Adv) on 3D point clouds to generate adversarial point clouds. First, we analyze the vulnerability of the 3D network model and extract the salient regions of the input point cloud, namely the vulnerable regions. Second, we propose an adaptive gradient attack algorithm that targets vulnerable regions. The proposed attack algorithm adaptively assigns different disturbances in different directions of the three-dimensional coordinates of the point cloud. Experimental results show that our proposed method AL-Adv achieves a higher attack success rate than the global attack method. Specifically, the adversarial examples generated by the AL-Adv demonstrate good imperceptibility and small generation costs.



## **12. Multi-metrics adaptively identifies backdoors in Federated learning**

cs.CR

13 pages, 8 figures and 6 tables

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06601v1) [paper-pdf](http://arxiv.org/pdf/2303.06601v1)

**Authors**: Siquan Huang, Yijiang Li, Chong Chen, Leyu Shi, Ying Gao

**Abstract**: The decentralized and privacy-preserving nature of federated learning (FL) makes it vulnerable to backdoor attacks aiming to manipulate the behavior of the resulting model on specific adversary-chosen inputs. However, most existing defenses based on statistical differences take effect only against specific attacks, especially when the malicious gradients are similar to benign ones or the data are highly non-independent and identically distributed (non-IID). In this paper, we revisit the distance-based defense methods and discover that i) Euclidean distance becomes meaningless in high dimensions and ii) malicious gradients with diverse characteristics cannot be identified by a single metric. To this end, we present a simple yet effective defense strategy with multi-metrics and dynamic weighting to identify backdoors adaptively. Furthermore, our novel defense has no reliance on predefined assumptions over attack settings or data distributions and little impact on benign performance. To evaluate the effectiveness of our approach, we conduct comprehensive experiments on different datasets under various attack settings, where our method achieves the best defensive performance. For instance, we achieve the lowest backdoor accuracy of 3.06% under the difficult Edge-case PGD, showing significant superiority over previous defenses. The results also demonstrate that our method can be well-adapted to a wide range of non-IID degrees without sacrificing the benign performance.



## **13. STPrivacy: Spatio-Temporal Privacy-Preserving Action Recognition**

cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2301.03046v2) [paper-pdf](http://arxiv.org/pdf/2301.03046v2)

**Authors**: Ming Li, Xiangyu Xu, Hehe Fan, Pan Zhou, Jun Liu, Jia-Wei Liu, Jiahe Li, Jussi Keppo, Mike Zheng Shou, Shuicheng Yan

**Abstract**: Existing methods of privacy-preserving action recognition (PPAR) mainly focus on frame-level (spatial) privacy removal through 2D CNNs. Unfortunately, they have two major drawbacks. First, they may compromise temporal dynamics in input videos, which are critical for accurate action recognition. Second, they are vulnerable to practical attacking scenarios where attackers probe for privacy from an entire video rather than individual frames. To address these issues, we propose a novel framework STPrivacy to perform video-level PPAR. For the first time, we introduce vision Transformers into PPAR by treating a video as a tubelet sequence, and accordingly design two complementary mechanisms, i.e., sparsification and anonymization, to remove privacy from a spatio-temporal perspective. In specific, our privacy sparsification mechanism applies adaptive token selection to abandon action-irrelevant tubelets. Then, our anonymization mechanism implicitly manipulates the remaining action-tubelets to erase privacy in the embedding space through adversarial learning. These mechanisms provide significant advantages in terms of privacy preservation for human eyes and action-privacy trade-off adjustment during deployment. We additionally contribute the first two large-scale PPAR benchmarks, VP-HMDB51 and VP-UCF101, to the community. Extensive evaluations on them, as well as two other tasks, validate the effectiveness and generalization capability of our framework.



## **14. Disclosure Risk from Homogeneity Attack in Differentially Private Frequency Distribution**

cs.CR

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2101.00311v5) [paper-pdf](http://arxiv.org/pdf/2101.00311v5)

**Authors**: Fang Liu, Xingyuan Zhao

**Abstract**: Differential privacy (DP) provides a robust model to achieve privacy guarantees for released information. We examine the protection potency of sanitized multi-dimensional frequency distributions via DP randomization mechanisms against homogeneity attack (HA). HA allows adversaries to obtain the exact values on sensitive attributes for their targets without having to identify them from the released data. We propose measures for disclosure risk from HA and derive closed-form relationships between the privacy loss parameters in DP and the disclosure risk from HA. The availability of the closed-form relationships assists understanding the abstract concepts of DP and privacy loss parameters by putting them in the context of a concrete privacy attack and offers a perspective for choosing privacy loss parameters when employing DP mechanisms in information sanitization and release in practice. We apply the closed-form mathematical relationships in real-life datasets to demonstrate the assessment of disclosure risk due to HA on differentially private sanitized frequency distributions at various privacy loss parameters.



## **15. Anomaly Detection with Ensemble of Encoder and Decoder**

cs.LG

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06431v1) [paper-pdf](http://arxiv.org/pdf/2303.06431v1)

**Authors**: Xijuan Sun, Di Wu, Arnaud Zinflou, Benoit Boulet

**Abstract**: Hacking and false data injection from adversaries can threaten power grids' everyday operations and cause significant economic loss. Anomaly detection in power grids aims to detect and discriminate anomalies caused by cyber attacks against the power system, which is essential for keeping power grids working correctly and efficiently. Different methods have been applied for anomaly detection, such as statistical methods and machine learning-based methods. Usually, machine learning-based methods need to model the normal data distribution. In this work, we propose a novel anomaly detection method by modeling the data distribution of normal samples via multiple encoders and decoders. Specifically, the proposed method maps input samples into a latent space and then reconstructs output samples from latent vectors. The extra encoder finally maps reconstructed samples to latent representations. During the training phase, we optimize parameters by minimizing the reconstruction loss and encoding loss. Training samples are re-weighted to focus more on missed correlations between features of normal data. Furthermore, we employ the long short-term memory model as encoders and decoders to test its effectiveness. We also investigate a meta-learning-based framework for hyper-parameter tuning of our approach. Experiment results on network intrusion and power system datasets demonstrate the effectiveness of our proposed method, where our models consistently outperform all baselines.



## **16. Improving the Robustness of Deep Convolutional Neural Networks Through Feature Learning**

cs.CV

8 pages, 12 figures, 6 tables. Work in process

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06425v1) [paper-pdf](http://arxiv.org/pdf/2303.06425v1)

**Authors**: Jin Ding, Jie-Chao Zhao, Yong-Zhi Sun, Ping Tan, Ji-En Ma, You-Tong Fang

**Abstract**: Deep convolutional neural network (DCNN for short) models are vulnerable to examples with small perturbations. Adversarial training (AT for short) is a widely used approach to enhance the robustness of DCNN models by data augmentation. In AT, the DCNN models are trained with clean examples and adversarial examples (AE for short) which are generated using a specific attack method, aiming to gain ability to defend themselves when facing the unseen AEs. However, in practice, the trained DCNN models are often fooled by the AEs generated by the novel attack methods. This naturally raises a question: can a DCNN model learn certain features which are insensitive to small perturbations, and further defend itself no matter what attack methods are presented. To answer this question, this paper makes a beginning effort by proposing a shallow binary feature module (SBFM for short), which can be integrated into any popular backbone. The SBFM includes two types of layers, i.e., Sobel layer and threshold layer. In Sobel layer, there are four parallel feature maps which represent horizontal, vertical, and diagonal edge features, respectively. And in threshold layer, it turns the edge features learnt by Sobel layer to the binary features, which then are feeded into the fully connected layers for classification with the features learnt by the backbone. We integrate SBFM into VGG16 and ResNet34, respectively, and conduct experiments on multiple datasets. Experimental results demonstrate, under FGSM attack with $\epsilon=8/255$, the SBFM integrated models can achieve averagely 35\% higher accuracy than the original ones, and in CIFAR-10 and TinyImageNet datasets, the SBFM integrated models can achieve averagely 75\% classification accuracy. The work in this paper shows it is promising to enhance the robustness of DCNN models through feature learning.



## **17. MorDIFF: Recognition Vulnerability and Attack Detectability of Face Morphing Attacks Created by Diffusion Autoencoders**

cs.CV

Accepted at the 11th International Workshop on Biometrics and  Forensics 2023 (IWBF 2023)

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2302.01843v2) [paper-pdf](http://arxiv.org/pdf/2302.01843v2)

**Authors**: Naser Damer, Meiling Fang, Patrick Siebke, Jan Niklas Kolf, Marco Huber, Fadi Boutros

**Abstract**: Investigating new methods of creating face morphing attacks is essential to foresee novel attacks and help mitigate them. Creating morphing attacks is commonly either performed on the image-level or on the representation-level. The representation-level morphing has been performed so far based on generative adversarial networks (GAN) where the encoded images are interpolated in the latent space to produce a morphed image based on the interpolated vector. Such a process was constrained by the limited reconstruction fidelity of GAN architectures. Recent advances in the diffusion autoencoder models have overcome the GAN limitations, leading to high reconstruction fidelity. This theoretically makes them a perfect candidate to perform representation-level face morphing. This work investigates using diffusion autoencoders to create face morphing attacks by comparing them to a wide range of image-level and representation-level morphs. Our vulnerability analyses on four state-of-the-art face recognition models have shown that such models are highly vulnerable to the created attacks, the MorDIFF, especially when compared to existing representation-level morphs. Detailed detectability analyses are also performed on the MorDIFF, showing that they are as challenging to detect as other morphing attacks created on the image- or representation-level. Data and morphing script are made public: https://github.com/naserdamer/MorDIFF.



## **18. Adversarial Attacks and Defenses in Machine Learning-Powered Networks: A Contemporary Survey**

cs.LG

46 pages, 21 figures

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06302v1) [paper-pdf](http://arxiv.org/pdf/2303.06302v1)

**Authors**: Yulong Wang, Tong Sun, Shenghong Li, Xin Yuan, Wei Ni, Ekram Hossain, H. Vincent Poor

**Abstract**: Adversarial attacks and defenses in machine learning and deep neural network have been gaining significant attention due to the rapidly growing applications of deep learning in the Internet and relevant scenarios. This survey provides a comprehensive overview of the recent advancements in the field of adversarial attack and defense techniques, with a focus on deep neural network-based classification models. Specifically, we conduct a comprehensive classification of recent adversarial attack methods and state-of-the-art adversarial defense techniques based on attack principles, and present them in visually appealing tables and tree diagrams. This is based on a rigorous evaluation of the existing works, including an analysis of their strengths and limitations. We also categorize the methods into counter-attack detection and robustness enhancement, with a specific focus on regularization-based methods for enhancing robustness. New avenues of attack are also explored, including search-based, decision-based, drop-based, and physical-world attacks, and a hierarchical classification of the latest defense methods is provided, highlighting the challenges of balancing training costs with performance, maintaining clean accuracy, overcoming the effect of gradient masking, and ensuring method transferability. At last, the lessons learned and open challenges are summarized with future research opportunities recommended.



## **19. Investigating Stateful Defenses Against Black-Box Adversarial Examples**

cs.CR

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06280v1) [paper-pdf](http://arxiv.org/pdf/2303.06280v1)

**Authors**: Ryan Feng, Ashish Hooda, Neal Mangaokar, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Defending machine-learning (ML) models against white-box adversarial attacks has proven to be extremely difficult. Instead, recent work has proposed stateful defenses in an attempt to defend against a more restricted black-box attacker. These defenses operate by tracking a history of incoming model queries, and rejecting those that are suspiciously similar. The current state-of-the-art stateful defense Blacklight was proposed at USENIX Security '22 and claims to prevent nearly 100% of attacks on both the CIFAR10 and ImageNet datasets. In this paper, we observe that an attacker can significantly reduce the accuracy of a Blacklight-protected classifier (e.g., from 82.2% to 6.4% on CIFAR10) by simply adjusting the parameters of an existing black-box attack. Motivated by this surprising observation, since existing attacks were evaluated by the Blacklight authors, we provide a systematization of stateful defenses to understand why existing stateful defense models fail. Finally, we propose a stronger evaluation strategy for stateful defenses comprised of adaptive score and hard-label based black-box attacks. We use these attacks to successfully reduce even reconfigured versions of Blacklight to as low as 0% robust accuracy.



## **20. Do we need entire training data for adversarial training?**

cs.CV

6 pages, 4 figures

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.06241v1) [paper-pdf](http://arxiv.org/pdf/2303.06241v1)

**Authors**: Vipul Gupta, Apurva Narayan

**Abstract**: Deep Neural Networks (DNNs) are being used to solve a wide range of problems in many domains including safety-critical domains like self-driving cars and medical imagery. DNNs suffer from vulnerability against adversarial attacks. In the past few years, numerous approaches have been proposed to tackle this problem by training networks using adversarial training. Almost all the approaches generate adversarial examples for the entire training dataset, thus increasing the training time drastically. We show that we can decrease the training time for any adversarial training algorithm by using only a subset of training data for adversarial training. To select the subset, we filter the adversarially-prone samples from the training data. We perform a simple adversarial attack on all training examples to filter this subset. In this attack, we add a small perturbation to each pixel and a few grid lines to the input image.   We perform adversarial training on the adversarially-prone subset and mix it with vanilla training performed on the entire dataset. Our results show that when our method-agnostic approach is plugged into FGSM, we achieve a speedup of 3.52x on MNIST and 1.98x on the CIFAR-10 dataset with comparable robust accuracy. We also test our approach on state-of-the-art Free adversarial training and achieve a speedup of 1.2x in training time with a marginal drop in robust accuracy on the ImageNet dataset.



## **21. Turning Strengths into Weaknesses: A Certified Robustness Inspired Attack Framework against Graph Neural Networks**

cs.CR

Accepted by CVPR 2023

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.06199v1) [paper-pdf](http://arxiv.org/pdf/2303.06199v1)

**Authors**: Binghui Wang, Meng Pang, Yun Dong

**Abstract**: Graph neural networks (GNNs) have achieved state-of-the-art performance in many graph learning tasks. However, recent studies show that GNNs are vulnerable to both test-time evasion and training-time poisoning attacks that perturb the graph structure. While existing attack methods have shown promising attack performance, we would like to design an attack framework to further enhance the performance. In particular, our attack framework is inspired by certified robustness, which was originally used by defenders to defend against adversarial attacks. We are the first, from the attacker perspective, to leverage its properties to better attack GNNs. Specifically, we first derive nodes' certified perturbation sizes against graph evasion and poisoning attacks based on randomized smoothing, respectively. A larger certified perturbation size of a node indicates this node is theoretically more robust to graph perturbations. Such a property motivates us to focus more on nodes with smaller certified perturbation sizes, as they are easier to be attacked after graph perturbations. Accordingly, we design a certified robustness inspired attack loss, when incorporated into (any) existing attacks, produces our certified robustness inspired attack counterpart. We apply our framework to the existing attacks and results show it can significantly enhance the existing base attacks' performance.



## **22. Harnessing the Speed and Accuracy of Machine Learning to Advance Cybersecurity**

cs.CR

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2302.12415v2) [paper-pdf](http://arxiv.org/pdf/2302.12415v2)

**Authors**: Khatoon Mohammed

**Abstract**: As cyber attacks continue to increase in frequency and sophistication, detecting malware has become a critical task for maintaining the security of computer systems. Traditional signature-based methods of malware detection have limitations in detecting complex and evolving threats. In recent years, machine learning (ML) has emerged as a promising solution to detect malware effectively. ML algorithms are capable of analyzing large datasets and identifying patterns that are difficult for humans to identify. This paper presents a comprehensive review of the state-of-the-art ML techniques used in malware detection, including supervised and unsupervised learning, deep learning, and reinforcement learning. We also examine the challenges and limitations of ML-based malware detection, such as the potential for adversarial attacks and the need for large amounts of labeled data. Furthermore, we discuss future directions in ML-based malware detection, including the integration of multiple ML algorithms and the use of explainable AI techniques to enhance the interpret ability of ML-based detection systems. Our research highlights the potential of ML-based techniques to improve the speed and accuracy of malware detection, and contribute to enhancing cybersecurity



## **23. Learning the Legibility of Visual Text Perturbations**

cs.CL

14 pages, 7 figures. Accepted at EACL 2023 (main, long)

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05077v2) [paper-pdf](http://arxiv.org/pdf/2303.05077v2)

**Authors**: Dev Seth, Rickard Stureborg, Danish Pruthi, Bhuwan Dhingra

**Abstract**: Many adversarial attacks in NLP perturb inputs to produce visually similar strings ('ergo' $\rightarrow$ '$\epsilon$rgo') which are legible to humans but degrade model performance. Although preserving legibility is a necessary condition for text perturbation, little work has been done to systematically characterize it; instead, legibility is typically loosely enforced via intuitions around the nature and extent of perturbations. Particularly, it is unclear to what extent can inputs be perturbed while preserving legibility, or how to quantify the legibility of a perturbed string. In this work, we address this gap by learning models that predict the legibility of a perturbed string, and rank candidate perturbations based on their legibility. To do so, we collect and release LEGIT, a human-annotated dataset comprising the legibility of visually perturbed text. Using this dataset, we build both text- and vision-based models which achieve up to $0.91$ F1 score in predicting whether an input is legible, and an accuracy of $0.86$ in predicting which of two given perturbations is more legible. Additionally, we discover that legible perturbations from the LEGIT dataset are more effective at lowering the performance of NLP models than best-known attack strategies, suggesting that current models may be vulnerable to a broad range of perturbations beyond what is captured by existing visual attacks. Data, code, and models are available at https://github.com/dvsth/learning-legibility-2023.



## **24. Exploring the Relationship between Architecture and Adversarially Robust Generalization**

cs.LG

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2209.14105v2) [paper-pdf](http://arxiv.org/pdf/2209.14105v2)

**Authors**: Aishan Liu, Shiyu Tang, Siyuan Liang, Ruihao Gong, Boxi Wu, Xianglong Liu, Dacheng Tao

**Abstract**: Adversarial training has been demonstrated to be one of the most effective remedies for defending adversarial examples, yet it often suffers from the huge robustness generalization gap on unseen testing adversaries, deemed as the adversarially robust generalization problem. Despite the preliminary understandings devoted to adversarially robust generalization, little is known from the architectural perspective. To bridge the gap, this paper for the first time systematically investigated the relationship between adversarially robust generalization and architectural design. Inparticular, we comprehensively evaluated 20 most representative adversarially trained architectures on ImageNette and CIFAR-10 datasets towards multiple `p-norm adversarial attacks. Based on the extensive experiments, we found that, under aligned settings, Vision Transformers (e.g., PVT, CoAtNet) often yield better adversarially robust generalization while CNNs tend to overfit on specific attacks and fail to generalize on multiple adversaries. To better understand the nature behind it, we conduct theoretical analysis via the lens of Rademacher complexity. We revealed the fact that the higher weight sparsity contributes significantly towards the better adversarially robust generalization of Transformers, which can be often achieved by the specially-designed attention blocks. We hope our paper could help to better understand the mechanism for designing robust DNNs. Our model weights can be found at http://robust.art.



## **25. Machine Learning Security in Industry: A Quantitative Survey**

cs.LG

Accepted at TIFS, version with more detailed appendix containing more  detailed statistical results. 17 pages, 6 tables and 4 figures

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2207.05164v2) [paper-pdf](http://arxiv.org/pdf/2207.05164v2)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Battista Biggio, Katharina Krombholz

**Abstract**: Despite the large body of academic work on machine learning security, little is known about the occurrence of attacks on machine learning systems in the wild. In this paper, we report on a quantitative study with 139 industrial practitioners. We analyze attack occurrence and concern and evaluate statistical hypotheses on factors influencing threat perception and exposure. Our results shed light on real-world attacks on deployed machine learning. On the organizational level, while we find no predictors for threat exposure in our sample, the amount of implement defenses depends on exposure to threats or expected likelihood to become a target. We also provide a detailed analysis of practitioners' replies on the relevance of individual machine learning attacks, unveiling complex concerns like unreliable decision making, business information leakage, and bias introduction into models. Finally, we find that on the individual level, prior knowledge about machine learning security influences threat perception. Our work paves the way for more research about adversarial machine learning in practice, but yields also insights for regulation and auditing.



## **26. TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets**

cs.LG

CVPR2023

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05762v1) [paper-pdf](http://arxiv.org/pdf/2303.05762v1)

**Authors**: Weixin Chen, Dawn Song, Bo Li

**Abstract**: Diffusion models have achieved great success in a range of tasks, such as image synthesis and molecule design. As such successes hinge on large-scale training data collected from diverse sources, the trustworthiness of these collected data is hard to control or audit. In this work, we aim to explore the vulnerabilities of diffusion models under potential training data manipulations and try to answer: How hard is it to perform Trojan attacks on well-trained diffusion models? What are the adversarial targets that such Trojan attacks can achieve? To answer these questions, we propose an effective Trojan attack against diffusion models, TrojDiff, which optimizes the Trojan diffusion and generative processes during training. In particular, we design novel transitions during the Trojan diffusion process to diffuse adversarial targets into a biased Gaussian distribution and propose a new parameterization of the Trojan generative process that leads to an effective training objective for the attack. In addition, we consider three types of adversarial targets: the Trojaned diffusion models will always output instances belonging to a certain class from the in-domain distribution (In-D2D attack), out-of-domain distribution (Out-D2D-attack), and one specific instance (D2I attack). We evaluate TrojDiff on CIFAR-10 and CelebA datasets against both DDPM and DDIM diffusion models. We show that TrojDiff always achieves high attack performance under different adversarial targets using different types of triggers, while the performance in benign environments is preserved. The code is available at https://github.com/chenweixin107/TrojDiff.



## **27. MIXPGD: Hybrid Adversarial Training for Speech Recognition Systems**

cs.SD

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05758v1) [paper-pdf](http://arxiv.org/pdf/2303.05758v1)

**Authors**: Aminul Huq, Weiyi Zhang, Xiaolin Hu

**Abstract**: Automatic speech recognition (ASR) systems based on deep neural networks are weak against adversarial perturbations. We propose mixPGD adversarial training method to improve the robustness of the model for ASR systems. In standard adversarial training, adversarial samples are generated by leveraging supervised or unsupervised methods. We merge the capabilities of both supervised and unsupervised approaches in our method to generate new adversarial samples which aid in improving model robustness. Extensive experiments and comparison across various state-of-the-art defense methods and adversarial attacks have been performed to show that mixPGD gains 4.1% WER of better performance than previous best performing models under white-box adversarial attack setting. We tested our proposed defense method against both white-box and transfer based black-box attack settings to ensure that our defense strategy is robust against various types of attacks. Empirical results on several adversarial attacks validate the effectiveness of our proposed approach.



## **28. Boosting Adversarial Attacks by Leveraging Decision Boundary Information**

cs.CV

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05719v1) [paper-pdf](http://arxiv.org/pdf/2303.05719v1)

**Authors**: Boheng Zeng, LianLi Gao, QiLong Zhang, ChaoQun Li, JingKuan Song, ShuaiQi Jing

**Abstract**: Due to the gap between a substitute model and a victim model, the gradient-based noise generated from a substitute model may have low transferability for a victim model since their gradients are different. Inspired by the fact that the decision boundaries of different models do not differ much, we conduct experiments and discover that the gradients of different models are more similar on the decision boundary than in the original position. Moreover, since the decision boundary in the vicinity of an input image is flat along most directions, we conjecture that the boundary gradients can help find an effective direction to cross the decision boundary of the victim models. Based on it, we propose a Boundary Fitting Attack to improve transferability. Specifically, we introduce a method to obtain a set of boundary points and leverage the gradient information of these points to update the adversarial examples. Notably, our method can be combined with existing gradient-based methods. Extensive experiments prove the effectiveness of our method, i.e., improving the success rate by 5.6% against normally trained CNNs and 14.9% against defense CNNs on average compared to state-of-the-art transfer-based attacks. Further we compare transformers with CNNs, the results indicate that transformers are more robust than CNNs. However, our method still outperforms existing methods when attacking transformers. Specifically, when using CNNs as substitute models, our method obtains an average attack success rate of 58.2%, which is 10.8% higher than other state-of-the-art transfer-based attacks.



## **29. On the Feasibility of Specialized Ability Stealing for Large Language Code Models**

cs.SE

11 pages

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.03012v2) [paper-pdf](http://arxiv.org/pdf/2303.03012v2)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao

**Abstract**: Recent progress in large language code models (LLCMs) has led to a dramatic surge in the use of software development. Nevertheless, it is widely known that training a well-performed LLCM requires a plethora of workforce for collecting the data and high quality annotation. Additionally, the training dataset may be proprietary (or partially open source to the public), and the training process is often conducted on a large-scale cluster of GPUs with high costs. Inspired by the recent success of imitation attacks in stealing computer vision and natural language models, this work launches the first imitation attack on LLCMs: by querying a target LLCM with carefully-designed queries and collecting the outputs, the adversary can train an imitation model that manifests close behavior with the target LLCM. We systematically investigate the effectiveness of launching imitation attacks under different query schemes and different LLCM tasks. We also design novel methods to polish the LLCM outputs, resulting in an effective imitation training process. We summarize our findings and provide lessons harvested in this study that can help better depict the attack surface of LLCMs. Our research contributes to the growing body of knowledge on imitation attacks and defenses in deep neural models, particularly in the domain of code related tasks.



## **30. NoiseCAM: Explainable AI for the Boundary Between Noise and Adversarial Attacks**

cs.LG

Submitted to IEEE Fuzzy 2023. arXiv admin note: text overlap with  arXiv:2303.06032

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.06151v1) [paper-pdf](http://arxiv.org/pdf/2303.06151v1)

**Authors**: Wenkai Tan, Justus Renkhoff, Alvaro Velasquez, Ziyu Wang, Lusi Li, Jian Wang, Shuteng Niu, Fan Yang, Yongxin Liu, Houbing Song

**Abstract**: Deep Learning (DL) and Deep Neural Networks (DNNs) are widely used in various domains. However, adversarial attacks can easily mislead a neural network and lead to wrong decisions. Defense mechanisms are highly preferred in safety-critical applications. In this paper, firstly, we use the gradient class activation map (GradCAM) to analyze the behavior deviation of the VGG-16 network when its inputs are mixed with adversarial perturbation or Gaussian noise. In particular, our method can locate vulnerable layers that are sensitive to adversarial perturbation and Gaussian noise. We also show that the behavior deviation of vulnerable layers can be used to detect adversarial examples. Secondly, we propose a novel NoiseCAM algorithm that integrates information from globally and pixel-level weighted class activation maps. Our algorithm is susceptible to adversarial perturbations and will not respond to Gaussian random noise mixed in the inputs. Third, we compare detecting adversarial examples using both behavior deviation and NoiseCAM, and we show that NoiseCAM outperforms behavior deviation modeling in its overall performance. Our work could provide a useful tool to defend against certain adversarial attacks on deep neural networks.



## **31. Evaluating the Robustness of Conversational Recommender Systems by Adversarial Examples**

cs.IR

10 pages

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.05575v1) [paper-pdf](http://arxiv.org/pdf/2303.05575v1)

**Authors**: Ali Montazeralghaem, James Allan

**Abstract**: Conversational recommender systems (CRSs) are improving rapidly, according to the standard recommendation accuracy metrics. However, it is essential to make sure that these systems are robust in interacting with users including regular and malicious users who want to attack the system by feeding the system modified input data. In this paper, we propose an adversarial evaluation scheme including four scenarios in two categories and automatically generate adversarial examples to evaluate the robustness of these systems in the face of different input data. By executing these adversarial examples we can compare the ability of different conversational recommender systems to satisfy the user's preferences. We evaluate three CRSs by the proposed adversarial examples on two datasets. Our results show that none of these systems are robust and reliable to the adversarial examples.



## **32. Efficient Certified Training and Robustness Verification of Neural ODEs**

cs.LG

Accepted at ICLR23

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.05246v1) [paper-pdf](http://arxiv.org/pdf/2303.05246v1)

**Authors**: Mustafa Zeqiri, Mark Niklas Müller, Marc Fischer, Martin Vechev

**Abstract**: Neural Ordinary Differential Equations (NODEs) are a novel neural architecture, built around initial value problems with learned dynamics which are solved during inference. Thought to be inherently more robust against adversarial perturbations, they were recently shown to be vulnerable to strong adversarial attacks, highlighting the need for formal guarantees. However, despite significant progress in robustness verification for standard feed-forward architectures, the verification of high dimensional NODEs remains an open problem. In this work, we address this challenge and propose GAINS, an analysis framework for NODEs combining three key ideas: (i) a novel class of ODE solvers, based on variable but discrete time steps, (ii) an efficient graph representation of solver trajectories, and (iii) a novel abstraction algorithm operating on this graph representation. Together, these advances enable the efficient analysis and certified training of high-dimensional NODEs, by reducing the runtime from an intractable $O(\exp(d)+\exp(T))$ to ${O}(d+T^2 \log^2T)$ in the dimensionality $d$ and integration time $T$. In an extensive evaluation on computer vision (MNIST and FMNIST) and time-series forecasting (PHYSIO-NET) problems, we demonstrate the effectiveness of both our certified training and verification methods.



## **33. Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors**

cs.CV

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.04238v2) [paper-pdf](http://arxiv.org/pdf/2303.04238v2)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called white-box attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. We show that our proposed method works both digitally and physically.



## **34. On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex**

cs.CL

Accepted at EACL2023 (main)

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2301.12868v3) [paper-pdf](http://arxiv.org/pdf/2301.12868v3)

**Authors**: Terry Yue Zhuo, Zhuang Li, Yujin Huang, Fatemeh Shiri, Weiqing Wang, Gholamreza Haffari, Yuan-Fang Li

**Abstract**: Semantic parsing is a technique aimed at constructing a structured representation of the meaning of a natural-language question. Recent advancements in few-shot language models trained on code have demonstrated superior performance in generating these representations compared to traditional unimodal language models, which are trained on downstream tasks. Despite these advancements, existing fine-tuned neural semantic parsers are susceptible to adversarial attacks on natural-language inputs. While it has been established that the robustness of smaller semantic parsers can be enhanced through adversarial training, this approach is not feasible for large language models in real-world scenarios, as it requires both substantial computational resources and expensive human annotation on in-domain semantic parsing data. This paper presents the first empirical study on the adversarial robustness of a large prompt-based language model of code, \codex. Our results demonstrate that the state-of-the-art (SOTA) code-language models are vulnerable to carefully crafted adversarial examples. To address this challenge, we propose methods for improving robustness without the need for significant amounts of labeled data or heavy computational resources.



## **35. Towards Good Practices in Evaluating Transfer Adversarial Attacks**

cs.CR

Our code and a list of categorized attacks are publicly available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2211.09565v2) [paper-pdf](http://arxiv.org/pdf/2211.09565v2)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes

**Abstract**: Transfer adversarial attacks raise critical security concerns in real-world, black-box scenarios. However, the actual progress of this field is difficult to assess due to two common limitations in existing evaluations. First, different methods are often not systematically and fairly evaluated in a one-to-one comparison. Second, only transferability is evaluated but another key attack property, stealthiness, is largely overlooked. In this work, we design good practices to address these limitations, and we present the first comprehensive evaluation of transfer attacks, covering 23 representative attacks against 9 defenses on ImageNet. In particular, we propose to categorize existing attacks into five categories, which enables our systematic category-wise analyses. These analyses lead to new findings that even challenge existing knowledge and also help determine the optimal attack hyperparameters for our attack-wise comprehensive evaluation. We also pay particular attention to stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics. Overall, our new insights into transferability and stealthiness lead to actionable good practices for future evaluations.



## **36. On the Robustness of Dataset Inference**

cs.LG

17 pages, 5 tables, 4 figures

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2210.13631v2) [paper-pdf](http://arxiv.org/pdf/2210.13631v2)

**Authors**: Sebastian Szyller, Rui Zhang, Jian Liu, N. Asokan

**Abstract**: Machine learning (ML) models are costly to train as they can require a significant amount of data, computational resources and technical expertise. Thus, they constitute valuable intellectual property that needs protection from adversaries wanting to steal them. Ownership verification techniques allow the victims of model stealing attacks to demonstrate that a suspect model was in fact stolen from theirs. Although a number of ownership verification techniques based on watermarking or fingerprinting have been proposed, most of them fall short either in terms of security guarantees (well-equipped adversaries can evade verification) or computational cost. A fingerprinting technique introduced at ICLR '21, Dataset Inference (DI), has been shown to offer better robustness and efficiency than prior methods. The authors of DI provided a correctness proof for linear (suspect) models. However, in the same setting, we prove that DI suffers from high false positives (FPs) -- it can incorrectly identify an independent model trained with non-overlapping data from the same distribution as stolen. We further prove that DI also triggers FPs in realistic, non-linear suspect models. We then confirm empirically that DI leads to FPs, with high confidence. Second, we show that DI also suffers from false negatives (FNs) -- an adversary can fool DI by regularising a stolen model's decision boundaries using adversarial training, thereby leading to an FN. To this end, we demonstrate that DI fails to identify a model adversarially trained from a stolen dataset -- the setting where DI is the hardest to evade. Finally, we discuss the implications of our findings, the viability of fingerprinting-based ownership verification in general, and suggest directions for future work.



## **37. Identification of Systematic Errors of Image Classifiers on Rare Subgroups**

cs.CV

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.05072v1) [paper-pdf](http://arxiv.org/pdf/2303.05072v1)

**Authors**: Jan Hendrik Metzen, Robin Hutmacher, N. Grace Hua, Valentyn Boreiko, Dan Zhang

**Abstract**: Despite excellent average-case performance of many image classifiers, their performance can substantially deteriorate on semantically coherent subgroups of the data that were under-represented in the training data. These systematic errors can impact both fairness for demographic minority groups as well as robustness and safety under domain shift. A major challenge is to identify such subgroups with subpar performance when the subgroups are not annotated and their occurrence is very rare. We leverage recent advances in text-to-image models and search in the space of textual descriptions of subgroups ("prompts") for subgroups where the target model has low performance on the prompt-conditioned synthesized data. To tackle the exponentially growing number of subgroups, we employ combinatorial testing. We denote this procedure as PromptAttack as it can be interpreted as an adversarial attack in a prompt space. We study subgroup coverage and identifiability with PromptAttack in a controlled setting and find that it identifies systematic errors with high accuracy. Thereupon, we apply PromptAttack to ImageNet classifiers and identify novel systematic errors on rare subgroups.



## **38. BeamAttack: Generating High-quality Textual Adversarial Examples through Beam Search and Mixed Semantic Spaces**

cs.CL

PAKDD2023

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.07199v1) [paper-pdf](http://arxiv.org/pdf/2303.07199v1)

**Authors**: Hai Zhu, Qingyang Zhao, Yuren Wu

**Abstract**: Natural language processing models based on neural networks are vulnerable to adversarial examples. These adversarial examples are imperceptible to human readers but can mislead models to make the wrong predictions. In a black-box setting, attacker can fool the model without knowing model's parameters and architecture. Previous works on word-level attacks widely use single semantic space and greedy search as a search strategy. However, these methods fail to balance the attack success rate, quality of adversarial examples and time consumption. In this paper, we propose BeamAttack, a textual attack algorithm that makes use of mixed semantic spaces and improved beam search to craft high-quality adversarial examples. Extensive experiments demonstrate that BeamAttack can improve attack success rate while saving numerous queries and time, e.g., improving at most 7\% attack success rate than greedy search when attacking the examples from MR dataset. Compared with heuristic search, BeamAttack can save at most 85\% model queries and achieve a competitive attack success rate. The adversarial examples crafted by BeamAttack are highly transferable and can effectively improve model's robustness during adversarial training. Code is available at https://github.com/zhuhai-ustc/beamattack/tree/master



## **39. On the Risks of Stealing the Decoding Algorithms of Language Models**

cs.LG

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.04729v2) [paper-pdf](http://arxiv.org/pdf/2303.04729v2)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2 and GPT-3. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.



## **40. Decision-BADGE: Decision-based Adversarial Batch Attack with Directional Gradient Estimation**

cs.CV

10 pages (8 pages except for references), 6 figures, 6 tables

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.04980v1) [paper-pdf](http://arxiv.org/pdf/2303.04980v1)

**Authors**: Geunhyeok Yu, Minwoo Jeon, Hyoseok Hwang

**Abstract**: The vulnerability of deep neural networks to adversarial examples has led to the rise in the use of adversarial attacks. While various decision-based and universal attack methods have been proposed, none have attempted to create a decision-based universal adversarial attack. This research proposes Decision-BADGE, which uses random gradient-free optimization and batch attack to generate universal adversarial perturbations for decision-based attacks. Multiple adversarial examples are combined to optimize a single universal perturbation, and the accuracy metric is reformulated into a continuous Hamming distance form. The effectiveness of accuracy metric as a loss function is demonstrated and mathematically proven. The combination of Decision-BADGE and the accuracy loss function performs better than both score-based image-dependent attack and white-box universal attack methods in terms of attack time efficiency. The research also shows that Decision-BADGE can successfully deceive unseen victims and accurately target specific classes.



## **41. Local Convolutions Cause an Implicit Bias towards High Frequency Adversarial Examples**

stat.ML

23 pages, 11 figures, 12 Tables

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2006.11440v5) [paper-pdf](http://arxiv.org/pdf/2006.11440v5)

**Authors**: Josue Ortega Caro, Yilong Ju, Ryan Pyle, Sourav Dey, Wieland Brendel, Fabio Anselmi, Ankit Patel

**Abstract**: Adversarial Attacks are still a significant challenge for neural networks. Recent work has shown that adversarial perturbations typically contain high-frequency features, but the root cause of this phenomenon remains unknown. Inspired by theoretical work on linear full-width convolutional models, we hypothesize that the local (i.e. bounded-width) convolutional operations commonly used in current neural networks are implicitly biased to learn high frequency features, and that this is one of the root causes of high frequency adversarial examples. To test this hypothesis, we analyzed the impact of different choices of linear and nonlinear architectures on the implicit bias of the learned features and the adversarial perturbations, in both spatial and frequency domains. We find that the high-frequency adversarial perturbations are critically dependent on the convolution operation because the spatially-limited nature of local convolutions induces an implicit bias towards high frequency features. The explanation for the latter involves the Fourier Uncertainty Principle: a spatially-limited (local in the space domain) filter cannot also be frequency-limited (local in the frequency domain). Furthermore, using larger convolution kernel sizes or avoiding convolutions (e.g. by using Vision Transformers architecture) significantly reduces this high frequency bias, but not the overall susceptibility to attacks. Looking forward, our work strongly suggests that understanding and controlling the implicit bias of architectures will be essential for achieving adversarial robustness.



## **42. Immune Defense: A Novel Adversarial Defense Mechanism for Preventing the Generation of Adversarial Examples**

cs.CV

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2303.04502v1) [paper-pdf](http://arxiv.org/pdf/2303.04502v1)

**Authors**: Jinwei Wang, Hao Wu, Haihua Wang, Jiawei Zhang, Xiangyang Luo, Bin Ma

**Abstract**: The vulnerability of Deep Neural Networks (DNNs) to adversarial examples has been confirmed. Existing adversarial defenses primarily aim at preventing adversarial examples from attacking DNNs successfully, rather than preventing their generation. If the generation of adversarial examples is unregulated, images within reach are no longer secure and pose a threat to non-robust DNNs. Although gradient obfuscation attempts to address this issue, it has been shown to be circumventable. Therefore, we propose a novel adversarial defense mechanism, which is referred to as immune defense and is the example-based pre-defense. This mechanism applies carefully designed quasi-imperceptible perturbations to the raw images to prevent the generation of adversarial examples for the raw images, and thereby protecting both images and DNNs. These perturbed images are referred to as Immune Examples (IEs). In the white-box immune defense, we provide a gradient-based and an optimization-based approach, respectively. Additionally, the more complex black-box immune defense is taken into consideration. We propose Masked Gradient Sign Descent (MGSD) to reduce approximation error and stabilize the update to improve the transferability of IEs and thereby ensure their effectiveness against black-box adversarial attacks. The experimental results demonstrate that the optimization-based approach has superior performance and better visual quality in white-box immune defense. In contrast, the gradient-based approach has stronger transferability and the proposed MGSD significantly improve the transferability of baselines.



## **43. Dishing Out DoS: How to Disable and Secure the Starlink User Terminal**

cs.CR

6 pages, 2 figures; the first two authors contributed equally to this  paper

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2303.00582v2) [paper-pdf](http://arxiv.org/pdf/2303.00582v2)

**Authors**: Joshua Smailes, Edd Salkield, Sebastian Köhler, Simon Birnbach, Ivan Martinovic

**Abstract**: Satellite user terminals are a promising target for adversaries seeking to target satellite communication networks. Despite this, many protections commonly found in terrestrial routers are not present in some user terminals.   As a case study we audit the attack surface presented by the Starlink router's admin interface, using fuzzing to uncover a denial of service attack on the Starlink user terminal. We explore the attack's impact, particularly in the cases of drive-by attackers, and attackers that are able to maintain a continuous presence on the network. Finally, we discuss wider implications, looking at lessons learned in terrestrial router security, and how to properly implement them in this new context.



## **44. GLOW: Global Layout Aware Attacks on Object Detection**

cs.CV

ICCV

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2302.14166v2) [paper-pdf](http://arxiv.org/pdf/2302.14166v2)

**Authors**: Buyu Liu, BaoJun, Jianping Fan, Xi Peng, Kui Ren, Jun Yu

**Abstract**: Adversarial attacks aim to perturb images such that a predictor outputs incorrect results. Due to the limited research in structured attacks, imposing consistency checks on natural multi-object scenes is a promising yet practical defense against conventional adversarial attacks. More desired attacks, to this end, should be able to fool defenses with such consistency checks. Therefore, we present the first approach GLOW that copes with various attack requests by generating global layout-aware adversarial attacks, in which both categorical and geometric layout constraints are explicitly established. Specifically, we focus on object detection task and given a victim image, GLOW first localizes victim objects according to target labels. And then it generates multiple attack plans, together with their context-consistency scores. Our proposed GLOW, on the one hand, is capable of handling various types of requests, including single or multiple victim objects, with or without specified victim objects. On the other hand, it produces a consistency score for each attack plan, reflecting the overall contextual consistency that both semantic category and global scene layout are considered. In experiment, we design multiple types of attack requests and validate our ideas on MS COCO and Pascal. Extensive experimental results demonstrate that we can achieve about 30$\%$ average relative improvement compared to state-of-the-art methods in conventional single object attack request; Moreover, our method outperforms SOTAs significantly on more generic attack requests by about 20$\%$ in average; Finally, our method produces superior performance under challenging zero-query black-box setting, or 20$\%$ better than SOTAs. Our code, model and attack requests would be made available.



## **45. Exploring Adversarial Attacks on Neural Networks: An Explainable Approach**

cs.LG

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2303.06032v1) [paper-pdf](http://arxiv.org/pdf/2303.06032v1)

**Authors**: Justus Renkhoff, Wenkai Tan, Alvaro Velasquez, illiam Yichen Wang, Yongxin Liu, Jian Wang, Shuteng Niu, Lejla Begic Fazlic, Guido Dartmann, Houbing Song

**Abstract**: Deep Learning (DL) is being applied in various domains, especially in safety-critical applications such as autonomous driving. Consequently, it is of great significance to ensure the robustness of these methods and thus counteract uncertain behaviors caused by adversarial attacks. In this paper, we use gradient heatmaps to analyze the response characteristics of the VGG-16 model when the input images are mixed with adversarial noise and statistically similar Gaussian random noise. In particular, we compare the network response layer by layer to determine where errors occurred. Several interesting findings are derived. First, compared to Gaussian random noise, intentionally generated adversarial noise causes severe behavior deviation by distracting the area of concentration in the networks. Second, in many cases, adversarial examples only need to compromise a few intermediate blocks to mislead the final decision. Third, our experiments revealed that specific blocks are more vulnerable and easier to exploit by adversarial examples. Finally, we demonstrate that the layers $Block4\_conv1$ and $Block5\_cov1$ of the VGG-16 model are more susceptible to adversarial attacks. Our work could provide valuable insights into developing more reliable Deep Neural Network (DNN) models.



## **46. Sampling Attacks on Meta Reinforcement Learning: A Minimax Formulation and Complexity Analysis**

cs.LG

updates: github repo posted

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2208.00081v2) [paper-pdf](http://arxiv.org/pdf/2208.00081v2)

**Authors**: Tao Li, Haozhe Lei, Quanyan Zhu

**Abstract**: Meta reinforcement learning (meta RL), as a combination of meta-learning ideas and reinforcement learning (RL), enables the agent to adapt to different tasks using a few samples. However, this sampling-based adaptation also makes meta RL vulnerable to adversarial attacks. By manipulating the reward feedback from sampling processes in meta RL, an attacker can mislead the agent into building wrong knowledge from training experience, which deteriorates the agent's performance when dealing with different tasks after adaptation. This paper provides a game-theoretical underpinning for understanding this type of security risk. In particular, we formally define the sampling attack model as a Stackelberg game between the attacker and the agent, which yields a minimax formulation. It leads to two online attack schemes: Intermittent Attack and Persistent Attack, which enable the attacker to learn an optimal sampling attack, defined by an $\epsilon$-first-order stationary point, within $\mathcal{O}(\epsilon^{-2})$ iterations. These attack schemes freeride the learning progress concurrently without extra interactions with the environment. By corroborating the convergence results with numerical experiments, we observe that a minor effort of the attacker can significantly deteriorate the learning performance, and the minimax approach can also help robustify the meta RL algorithms.



## **47. Robustness-preserving Lifelong Learning via Dataset Condensation**

cs.LG

Accepted by ICASSP2023 Main Track: Machine Learning for Signal  Processing

**SubmitDate**: 2023-03-07    [abs](http://arxiv.org/abs/2303.04183v1) [paper-pdf](http://arxiv.org/pdf/2303.04183v1)

**Authors**: Jinghan Jia, Yihua Zhang, Dogyoon Song, Sijia Liu, Alfred Hero

**Abstract**: Lifelong learning (LL) aims to improve a predictive model as the data source evolves continuously. Most work in this learning paradigm has focused on resolving the problem of 'catastrophic forgetting,' which refers to a notorious dilemma between improving model accuracy over new data and retaining accuracy over previous data. Yet, it is also known that machine learning (ML) models can be vulnerable in the sense that tiny, adversarial input perturbations can deceive the models into producing erroneous predictions. This motivates the research objective of this paper - specification of a new LL framework that can salvage model robustness (against adversarial attacks) from catastrophic forgetting. Specifically, we propose a new memory-replay LL strategy that leverages modern bi-level optimization techniques to determine the 'coreset' of the current data (i.e., a small amount of data to be memorized) for ease of preserving adversarial robustness over time. We term the resulting LL framework 'Data-Efficient Robustness-Preserving LL' (DERPLL). The effectiveness of DERPLL is evaluated for class-incremental image classification using ResNet-18 over the CIFAR-10 dataset. Experimental results show that DERPLL outperforms the conventional coreset-guided LL baseline and achieves a substantial improvement in both standard accuracy and robust accuracy.



## **48. Exploiting Trust for Resilient Hypothesis Testing with Malicious Robots (evolved version)**

cs.RO

21 pages, 5 figures, 1 table. arXiv admin note: substantial text  overlap with arXiv:2209.12285

**SubmitDate**: 2023-03-07    [abs](http://arxiv.org/abs/2303.04075v1) [paper-pdf](http://arxiv.org/pdf/2303.04075v1)

**Authors**: Matthew Cavorsi, Orhan Eren Akgün, Michal Yemini, Andrea Goldsmith, Stephanie Gil

**Abstract**: We develop a resilient binary hypothesis testing framework for decision making in adversarial multi-robot crowdsensing tasks. This framework exploits stochastic trust observations between robots to arrive at tractable, resilient decision making at a centralized Fusion Center (FC) even when i) there exist malicious robots in the network and their number may be larger than the number of legitimate robots, and ii) the FC uses one-shot noisy measurements from all robots. We derive two algorithms to achieve this. The first is the Two Stage Approach (2SA) that estimates the legitimacy of robots based on received trust observations, and provably minimizes the probability of detection error in the worst-case malicious attack. Here, the proportion of malicious robots is known but arbitrary. For the case of an unknown proportion of malicious robots, we develop the Adversarial Generalized Likelihood Ratio Test (A-GLRT) that uses both the reported robot measurements and trust observations to estimate the trustworthiness of robots, their reporting strategy, and the correct hypothesis simultaneously. We exploit special problem structure to show that this approach remains computationally tractable despite several unknown problem parameters. We deploy both algorithms in a hardware experiment where a group of robots conducts crowdsensing of traffic conditions on a mock-up road network similar in spirit to Google Maps, subject to a Sybil attack. We extract the trust observations for each robot from actual communication signals which provide statistical information on the uniqueness of the sender. We show that even when the malicious robots are in the majority, the FC can reduce the probability of detection error to 30.5% and 29% for the 2SA and the A-GLRT respectively.



## **49. Bounding Information Leakage in Machine Learning**

cs.LG

Published in [Elsevier  Neurocomputing](https://doi.org/10.1016/j.neucom.2023.02.058)

**SubmitDate**: 2023-03-07    [abs](http://arxiv.org/abs/2105.03875v2) [paper-pdf](http://arxiv.org/pdf/2105.03875v2)

**Authors**: Ganesh Del Grosso, Georg Pichler, Catuscia Palamidessi, Pablo Piantanida

**Abstract**: Recently, it has been shown that Machine Learning models can leak sensitive information about their training data. This information leakage is exposed through membership and attribute inference attacks. Although many attack strategies have been proposed, little effort has been made to formalize these problems. We present a novel formalism, generalizing membership and attribute inference attack setups previously studied in the literature and connecting them to memorization and generalization. First, we derive a universal bound on the success rate of inference attacks and connect it to the generalization gap of the target model. Second, we study the question of how much sensitive information is stored by the algorithm about its training set and we derive bounds on the mutual information between the sensitive attributes and model parameters. Experimentally, we illustrate the potential of our approach by applying it to both synthetic data and classification tasks on natural images. Finally, we apply our formalism to different attribute inference strategies, with which an adversary is able to recover the identity of writers in the PenDigits dataset.



## **50. SCRAMBLE-CFI: Mitigating Fault-Induced Control-Flow Attacks on OpenTitan**

cs.CR

**SubmitDate**: 2023-03-07    [abs](http://arxiv.org/abs/2303.03711v1) [paper-pdf](http://arxiv.org/pdf/2303.03711v1)

**Authors**: Pascal Nasahl, Stefan Mangard

**Abstract**: Secure elements physically exposed to adversaries are frequently targeted by fault attacks. These attacks can be utilized to hijack the control-flow of software allowing the attacker to bypass security measures, extract sensitive data, or gain full code execution. In this paper, we systematically analyze the threat vector of fault-induced control-flow manipulations on the open-source OpenTitan secure element. Our thorough analysis reveals that current countermeasures of this chip either induce large area overheads or still cannot prevent the attacker from exploiting the identified threats. In this context, we introduce SCRAMBLE-CFI, an encryption-based control-flow integrity scheme utilizing existing hardware features of OpenTitan. SCRAMBLE-CFI confines, with minimal hardware overhead, the impact of fault-induced control-flow attacks by encrypting each function with a different encryption tweak at load-time. At runtime, code only can be successfully decrypted when the correct decryption tweak is active. We open-source our hardware changes and release our LLVM toolchain automatically protecting programs. Our analysis shows that SCRAMBLE-CFI complementarily enhances security guarantees of OpenTitan with a negligible hardware overhead of less than 3.97 % and a runtime overhead of 7.02 % for the Embench-IoT benchmarks.



