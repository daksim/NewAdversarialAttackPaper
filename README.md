# Latest Adversarial Attack Papers
**update at 2022-06-03 06:31:44**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adversarial Attacks on Gaussian Process Bandits**

stat.ML

Accepted to ICML 2022

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2110.08449v2)

**Authors**: Eric Han, Jonathan Scarlett

**Abstracts**: Gaussian processes (GP) are a widely-adopted tool used to sequentially optimize black-box functions, where evaluations are costly and potentially noisy. Recent works on GP bandits have proposed to move beyond random noise and devise algorithms robust to adversarial attacks. This paper studies this problem from the attacker's perspective, proposing various adversarial attack methods with differing assumptions on the attacker's strength and prior information. Our goal is to understand adversarial attacks on GP bandits from theoretical and practical perspectives. We focus primarily on targeted attacks on the popular GP-UCB algorithm and a related elimination-based algorithm, based on adversarially perturbing the function $f$ to produce another function $\tilde{f}$ whose optima are in some target region $\mathcal{R}_{\rm target}$. Based on our theoretical analysis, we devise both white-box attacks (known $f$) and black-box attacks (unknown $f$), with the former including a Subtraction attack and Clipping attack, and the latter including an Aggressive subtraction attack. We demonstrate that adversarial attacks on GP bandits can succeed in forcing the algorithm towards $\mathcal{R}_{\rm target}$ even with a low attack budget, and we test our attacks' effectiveness on a diverse range of objective functions.



## **2. The robust way to stack and bag: the local Lipschitz way**

cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00513v1)

**Authors**: Thulasi Tholeti, Sheetal Kalyani

**Abstracts**: Recent research has established that the local Lipschitz constant of a neural network directly influences its adversarial robustness. We exploit this relationship to construct an ensemble of neural networks which not only improves the accuracy, but also provides increased adversarial robustness. The local Lipschitz constants for two different ensemble methods - bagging and stacking - are derived and the architectures best suited for ensuring adversarial robustness are deduced. The proposed ensemble architectures are tested on MNIST and CIFAR-10 datasets in the presence of white-box attacks, FGSM and PGD. The proposed architecture is found to be more robust than a) a single network and b) traditional ensemble methods.



## **3. Attack-Agnostic Adversarial Detection**

cs.CV

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00489v1)

**Authors**: Jiaxin Cheng, Mohamed Hussein, Jay Billa, Wael AbdAlmageed

**Abstracts**: The growing number of adversarial attacks in recent years gives attackers an advantage over defenders, as defenders must train detectors after knowing the types of attacks, and many models need to be maintained to ensure good performance in detecting any upcoming attacks. We propose a way to end the tug-of-war between attackers and defenders by treating adversarial attack detection as an anomaly detection problem so that the detector is agnostic to the attack. We quantify the statistical deviation caused by adversarial perturbations in two aspects. The Least Significant Component Feature (LSCF) quantifies the deviation of adversarial examples from the statistics of benign samples and Hessian Feature (HF) reflects how adversarial examples distort the landscape of the model's optima by measuring the local loss curvature. Empirical results show that our method can achieve an overall ROC AUC of 94.9%, 89.7%, and 94.6% on CIFAR10, CIFAR100, and SVHN, respectively, and has comparable performance to adversarial detectors trained with adversarial examples on most of the attacks.



## **4. Generating End-to-End Adversarial Examples for Malware Classifiers Using Explainability**

cs.CR

Accepted as a conference paper at IJCNN 2020

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2009.13243v2)

**Authors**: Ishai Rosenberg, Shai Meir, Jonathan Berrebi, Ilay Gordon, Guillaume Sicard, Eli David

**Abstracts**: In recent years, the topic of explainable machine learning (ML) has been extensively researched. Up until now, this research focused on regular ML users use-cases such as debugging a ML model. This paper takes a different posture and show that adversaries can leverage explainable ML to bypass multi-feature types malware classifiers. Previous adversarial attacks against such classifiers only add new features and not modify existing ones to avoid harming the modified malware executable's functionality. Current attacks use a single algorithm that both selects which features to modify and modifies them blindly, treating all features the same. In this paper, we present a different approach. We split the adversarial example generation task into two parts: First we find the importance of all features for a specific sample using explainability algorithms, and then we conduct a feature-specific modification, feature-by-feature. In order to apply our attack in black-box scenarios, we introduce the concept of transferability of explainability, that is, applying explainability algorithms to different classifiers using different features subsets and trained on different datasets still result in a similar subset of important features. We conclude that explainability algorithms can be leveraged by adversaries and thus the advocates of training more interpretable classifiers should consider the trade-off of higher vulnerability of those classifiers to adversarial attacks.



## **5. Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations**

cs.CR

Accepted by IJCAI 2022

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00477v1)

**Authors**: Run Wang, Ziheng Huang, Zhikai Chen, Li Liu, Jing Chen, Lina Wang

**Abstracts**: DeepFake is becoming a real risk to society and brings potential threats to both individual privacy and political security due to the DeepFaked multimedia are realistic and convincing. However, the popular DeepFake passive detection is an ex-post forensics countermeasure and failed in blocking the disinformation spreading in advance. To address this limitation, researchers study the proactive defense techniques by adding adversarial noises into the source data to disrupt the DeepFake manipulation. However, the existing studies on proactive DeepFake defense via injecting adversarial noises are not robust, which could be easily bypassed by employing simple image reconstruction revealed in a recent study MagDR.   In this paper, we investigate the vulnerability of the existing forgery techniques and propose a novel \emph{anti-forgery} technique that helps users protect the shared facial images from attackers who are capable of applying the popular forgery techniques. Our proposed method generates perceptual-aware perturbations in an incessant manner which is vastly different from the prior studies by adding adversarial noises that is sparse. Experimental results reveal that our perceptual-aware perturbations are robust to diverse image transformations, especially the competitive evasion technique, MagDR via image reconstruction. Our findings potentially open up a new research direction towards thorough understanding and investigation of perceptual-aware adversarial attack for protecting facial images against DeepFakes in a proactive and robust manner. We open-source our tool to foster future research. Code is available at https://github.com/AbstractTeen/AntiForgery/.



## **6. PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations**

cs.CR

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2205.13523v2)

**Authors**: Manaar Alam, Esha Sarkar, Michail Maniatakos

**Abstracts**: Federated Learning (FL) enables numerous participants to train deep learning models collaboratively without exposing their personal, potentially sensitive data, making it a promising solution for data privacy in collaborative training. The distributed nature of FL and unvetted data, however, makes it inherently vulnerable to backdoor attacks: In this scenario, an adversary injects backdoor functionality into the centralized model during training, which can be triggered to cause the desired misclassification for a specific adversary-chosen input. A range of prior work establishes successful backdoor injection in an FL system; however, these backdoors are not demonstrated to be long-lasting. The backdoor functionality does not remain in the system if the adversary is removed from the training process since the centralized model parameters continuously mutate during successive FL training rounds. Therefore, in this work, we propose PerDoor, a persistent-by-construction backdoor injection technique for FL, driven by adversarial perturbation and targeting parameters of the centralized model that deviate less in successive FL rounds and contribute the least to the main task accuracy. An exhaustive evaluation considering an image classification scenario portrays on average $10.5\times$ persistence over multiple FL rounds compared to traditional backdoor attacks. Through experiments, we further exhibit the potency of PerDoor in the presence of state-of-the-art backdoor prevention techniques in an FL system. Additionally, the operation of adversarial perturbation also assists PerDoor in developing non-uniform trigger patterns for backdoor inputs compared to uniform triggers (with fixed patterns and locations) of existing backdoor techniques, which are prone to be easily mitigated.



## **7. NeuroUnlock: Unlocking the Architecture of Obfuscated Deep Neural Networks**

cs.CR

The definitive Version of Record will be Published in the 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00402v1)

**Authors**: Mahya Morid Ahmadi, Lilas Alrahis, Alessio Colucci, Ozgur Sinanoglu, Muhammad Shafique

**Abstracts**: The advancements of deep neural networks (DNNs) have led to their deployment in diverse settings, including safety and security-critical applications. As a result, the characteristics of these models have become sensitive intellectual properties that require protection from malicious users. Extracting the architecture of a DNN through leaky side-channels (e.g., memory access) allows adversaries to (i) clone the model, and (ii) craft adversarial attacks. DNN obfuscation thwarts side-channel-based architecture stealing (SCAS) attacks by altering the run-time traces of a given DNN while preserving its functionality. In this work, we expose the vulnerability of state-of-the-art DNN obfuscation methods to these attacks. We present NeuroUnlock, a novel SCAS attack against obfuscated DNNs. Our NeuroUnlock employs a sequence-to-sequence model that learns the obfuscation procedure and automatically reverts it, thereby recovering the original DNN architecture. We demonstrate the effectiveness of NeuroUnlock by recovering the architecture of 200 randomly generated and obfuscated DNNs running on the Nvidia RTX 2080 TI graphics processing unit (GPU). Moreover, NeuroUnlock recovers the architecture of various other obfuscated DNNs, such as the VGG-11, VGG-13, ResNet-20, and ResNet-32 networks. After recovering the architecture, NeuroUnlock automatically builds a near-equivalent DNN with only a 1.4% drop in the testing accuracy. We further show that launching a subsequent adversarial attack on the recovered DNNs boosts the success rate of the adversarial attack by 51.7% in average compared to launching it on the obfuscated versions. Additionally, we propose a novel methodology for DNN obfuscation, ReDLock, which eradicates the deterministic nature of the obfuscation and achieves 2.16X more resilience to the NeuroUnlock attack. We release the NeuroUnlock and the ReDLock as open-source frameworks.



## **8. Support Vector Machines under Adversarial Label Contamination**

cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00352v1)

**Authors**: Huang Xiao, Battista Biggio, Blaine Nelson, Han Xiao, Claudia Eckert, Fabio Roli

**Abstracts**: Machine learning algorithms are increasingly being applied in security-related tasks such as spam and malware detection, although their security properties against deliberate attacks have not yet been widely understood. Intelligent and adaptive attackers may indeed exploit specific vulnerabilities exposed by machine learning techniques to violate system security. Being robust to adversarial data manipulation is thus an important, additional requirement for machine learning algorithms to successfully operate in adversarial settings. In this work, we evaluate the security of Support Vector Machines (SVMs) to well-crafted, adversarial label noise attacks. In particular, we consider an attacker that aims to maximize the SVM's classification error by flipping a number of labels in the training data. We formalize a corresponding optimal attack strategy, and solve it by means of heuristic approaches to keep the computational complexity tractable. We report an extensive experimental analysis on the effectiveness of the considered attacks against linear and non-linear SVMs, both on synthetic and real-world datasets. We finally argue that our approach can also provide useful insights for developing more secure SVM learning algorithms, and also novel techniques in a number of related research areas, such as semi-supervised and active learning.



## **9. A Simple Structure For Building A Robust Model**

cs.CV

Accepted by Fifth International Conference on Intelligence Science  (ICIS2022); 10 pages, 3 figures, 4 tables

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2204.11596v2)

**Authors**: Xiao Tan, Jingbo Gao, Ruolin Li

**Abstracts**: As deep learning applications, especially programs of computer vision, are increasingly deployed in our lives, we have to think more urgently about the security of these applications.One effective way to improve the security of deep learning models is to perform adversarial training, which allows the model to be compatible with samples that are deliberately created for use in attacking the model.Based on this, we propose a simple architecture to build a model with a certain degree of robustness, which improves the robustness of the trained network by adding an adversarial sample detection network for cooperative training. At the same time, we design a new data sampling strategy that incorporates multiple existing attacks, allowing the model to adapt to many different adversarial attacks with a single training.We conducted some experiments to test the effectiveness of this design based on Cifar10 dataset, and the results indicate that it has some degree of positive effect on the robustness of the model.Our code could be found at https://github.com/dowdyboy/simple_structure_for_robust_model .



## **10. Bounding Membership Inference**

cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2202.12232v2)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $\epsilon$-DP or $(\epsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.



## **11. Metamorphic Testing-based Adversarial Attack to Fool Deepfake Detectors**

cs.CV

paper accepted at 26TH International Conference on Pattern  Recognition (ICPR2022)

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2204.08612v2)

**Authors**: Nyee Thoang Lim, Meng Yi Kuan, Muxin Pu, Mei Kuan Lim, Chun Yong Chong

**Abstracts**: Deepfakes utilise Artificial Intelligence (AI) techniques to create synthetic media where the likeness of one person is replaced with another. There are growing concerns that deepfakes can be maliciously used to create misleading and harmful digital contents. As deepfakes become more common, there is a dire need for deepfake detection technology to help spot deepfake media. Present deepfake detection models are able to achieve outstanding accuracy (>90%). However, most of them are limited to within-dataset scenario, where the same dataset is used for training and testing. Most models do not generalise well enough in cross-dataset scenario, where models are tested on unseen datasets from another source. Furthermore, state-of-the-art deepfake detection models rely on neural network-based classification models that are known to be vulnerable to adversarial attacks. Motivated by the need for a robust deepfake detection model, this study adapts metamorphic testing (MT) principles to help identify potential factors that could influence the robustness of the examined model, while overcoming the test oracle problem in this domain. Metamorphic testing is specifically chosen as the testing technique as it fits our demand to address learning-based system testing with probabilistic outcomes from largely black-box components, based on potentially large input domains. We performed our evaluations on MesoInception-4 and TwoStreamNet models, which are the state-of-the-art deepfake detection models. This study identified makeup application as an adversarial attack that could fool deepfake detectors. Our experimental results demonstrate that both the MesoInception-4 and TwoStreamNet models degrade in their performance by up to 30\% when the input data is perturbed with makeup.



## **12. FoveaTer: Foveated Transformer for Image Classification**

cs.CV

17 pages, 7 figures

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2105.14173v2)

**Authors**: Aditya Jonnalagadda, William Yang Wang, B. S. Manjunath, Miguel P. Eckstein

**Abstracts**: Many animals and humans process the visual field with a varying spatial resolution (foveated vision) and use peripheral processing to make eye movements and point the fovea to acquire high-resolution information about objects of interest. This architecture results in computationally efficient rapid scene exploration. Recent progress in self-attention-based vision Transformers allow global interactions between feature locations and result in increased robustness to adversarial attacks. However, the Transformer models do not explicitly model the foveated properties of the visual system nor the interaction between eye movements and the classification task. We propose foveated Transformer (FoveaTer) model, which uses pooling regions and eye movements to perform object classification tasks. Our proposed model pools the image features using squared pooling regions, an approximation to the biologically-inspired foveated architecture. It decides on subsequent fixation locations based on the attention assigned by the Transformer to various locations from past and present fixations. It dynamically allocates more fixation/computational resources to more challenging images before making the final object category decision. We compare FoveaTer against a Full-resolution baseline model, which does not contain any pooling. On the ImageNet dataset, the Foveated model with Dynamic-stop achieves an accuracy of $1.9\%$ below the full-resolution model with a throughput gain of $51\%$. Using a Foveated model with Dynamic-stop and the Full-resolution model, the ensemble outperforms the baseline Full-resolution by $0.2\%$ with a throughput gain of $7.7\%$. We also demonstrate our model's robustness against adversarial attacks. Finally, we compare the Foveated model to human performance in a scene categorization task and show similar dependence of accuracy with number of exploratory fixations.



## **13. Generative Models with Information-Theoretic Protection Against Membership Inference Attacks**

cs.LG

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2206.00071v1)

**Authors**: Parisa Hassanzadeh, Robert E. Tillman

**Abstracts**: Deep generative models, such as Generative Adversarial Networks (GANs), synthesize diverse high-fidelity data samples by estimating the underlying distribution of high dimensional data. Despite their success, GANs may disclose private information from the data they are trained on, making them susceptible to adversarial attacks such as membership inference attacks, in which an adversary aims to determine if a record was part of the training set. We propose an information theoretically motivated regularization term that prevents the generative model from overfitting to training data and encourages generalizability. We show that this penalty minimizes the JensenShannon divergence between components of the generator trained on data with different membership, and that it can be implemented at low cost using an additional classifier. Our experiments on image datasets demonstrate that with the proposed regularization, which comes at only a small added computational cost, GANs are able to preserve privacy and generate high-quality samples that achieve better downstream classification performance compared to non-private and differentially private generative models.



## **14. CodeAttack: Code-based Adversarial Attacks for Pre-Trained Programming Language Models**

cs.CL

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2206.00052v1)

**Authors**: Akshita Jha, Chandan K. Reddy

**Abstracts**: Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks. We propose, CodeAttack, a simple yet effective black-box attack model that uses code structure to generate imperceptible, effective, and minimally perturbed adversarial code samples. We demonstrate the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall performance while being more efficient and imperceptible.



## **15. Hide and Seek: on the Stealthiness of Attacks against Deep Learning Systems**

cs.CR

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2205.15944v1)

**Authors**: Zeyan Liu, Fengjun Li, Jingqiang Lin, Zhu Li, Bo Luo

**Abstracts**: With the growing popularity of artificial intelligence and machine learning, a wide spectrum of attacks against deep learning models have been proposed in the literature. Both the evasion attacks and the poisoning attacks attempt to utilize adversarially altered samples to fool the victim model to misclassify the adversarial sample. While such attacks claim to be or are expected to be stealthy, i.e., imperceptible to human eyes, such claims are rarely evaluated. In this paper, we present the first large-scale study on the stealthiness of adversarial samples used in the attacks against deep learning. We have implemented 20 representative adversarial ML attacks on six popular benchmarking datasets. We evaluate the stealthiness of the attack samples using two complementary approaches: (1) a numerical study that adopts 24 metrics for image similarity or quality assessment; and (2) a user study of 3 sets of questionnaires that has collected 20,000+ annotations from 1,000+ responses. Our results show that the majority of the existing attacks introduce nonnegligible perturbations that are not stealthy to human eyes. We further analyze the factors that contribute to attack stealthiness. We further examine the correlation between the numerical analysis and the user studies, and demonstrate that some image quality metrics may provide useful guidance in attack designs, while there is still a significant gap between assessed image quality and visual stealthiness of attacks.



## **16. Atomic cross-chain exchanges of shared assets**

cs.CR

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2202.12855v2)

**Authors**: Krishnasuri Narayanam, Venkatraman Ramakrishna, Dhinakaran Vinayagamurthy, Sandeep Nishad

**Abstracts**: A core enabler for blockchain or DLT interoperability is the ability to atomically exchange assets held by mutually untrusting owners on different ledgers. This atomic swap problem has been well-studied, with the Hash Time Locked Contract (HTLC) emerging as a canonical solution. HTLC ensures atomicity of exchange, albeit with caveats for node failure and timeliness of claims. But a bigger limitation of HTLC is that it only applies to a model consisting of two adversarial parties having sole ownership of a single asset in each ledger. Realistic extensions of the model in which assets may be jointly owned by multiple parties, all of whose consents are required for exchanges, or where multiple assets must be exchanged for one, are susceptible to collusion attacks and hence cannot be handled by HTLC. In this paper, we generalize the model of asset exchanges across DLT networks and present a taxonomy of use cases, describe the threat model, and propose MPHTLC, an augmented HTLC protocol for atomic multi-owner-and-asset exchanges. We analyze the correctness, safety, and application scope of MPHTLC. As proof-of-concept, we show how MPHTLC primitives can be implemented in networks built on Hyperledger Fabric and Corda, and how MPHTLC can be implemented in the Hyperledger Labs Weaver framework by augmenting its existing HTLC protocol.



## **17. Semantic Autoencoder and Its Potential Usage for Adversarial Attack**

cs.LG

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2205.15592v1)

**Authors**: Yurui Ming, Cuihuan Du, Chin-Teng Lin

**Abstracts**: Autoencoder can give rise to an appropriate latent representation of the input data, however, the representation which is solely based on the intrinsic property of the input data, is usually inferior to express some semantic information. A typical case is the potential incapability of forming a clear boundary upon clustering of these representations. By encoding the latent representation that not only depends on the content of the input data, but also the semantic of the input data, such as label information, we propose an enhanced autoencoder architecture named semantic autoencoder. Experiments of representation distribution via t-SNE shows a clear distinction between these two types of encoders and confirm the supremacy of the semantic one, whilst the decoded samples of these two types of autoencoders exhibit faint dissimilarity either objectively or subjectively. Based on this observation, we consider adversarial attacks to learning algorithms that rely on the latent representation obtained via autoencoders. It turns out that latent contents of adversarial samples constructed from semantic encoder with deliberate wrong label information exhibit different distribution compared with that of the original input data, while both of these samples manifest very marginal difference. This new way of attack set up by our work is worthy of attention due to the necessity to secure the widespread deep learning applications.



## **18. Connecting adversarial attacks and optimal transport for domain adaptation**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15424v1)

**Authors**: Arip Asadulaev, Vitaly Shutov, Alexander Korotin, Alexander Panfilov, Andrey Filchenkov

**Abstracts**: We present a novel algorithm for domain adaptation using optimal transport. In domain adaptation, the goal is to adapt a classifier trained on the source domain samples to the target domain. In our method, we use optimal transport to map target samples to the domain named source fiction. This domain differs from the source but is accurately classified by the source domain classifier. Our main idea is to generate a source fiction by c-cyclically monotone transformation over the target domain. If samples with the same labels in two domains are c-cyclically monotone, the optimal transport map between these domains preserves the class-wise structure, which is the main goal of domain adaptation. To generate a source fiction domain, we propose an algorithm that is based on our finding that adversarial attacks are a c-cyclically monotone transformation of the dataset. We conduct experiments on Digits and Modern Office-31 datasets and achieve improvement in performance for simple discrete optimal transport solvers for all adaptation tasks.



## **19. Fooling SHAP with Stealthily Biased Sampling**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15419v1)

**Authors**: Gabriel Laberge, Ulrich Aïvodji, Satoshi Hara

**Abstracts**: SHAP explanations aim at identifying which features contribute the most to the difference in model prediction at a specific input versus a background distribution. Recent studies have shown that they can be manipulated by malicious adversaries to produce arbitrary desired explanations. However, existing attacks focus solely on altering the black-box model itself. In this paper, we propose a complementary family of attacks that leave the model intact and manipulate SHAP explanations using stealthily biased sampling of the data points used to approximate expectations w.r.t the background distribution. In the context of fairness audit, we show that our attack can reduce the importance of a sensitive feature when explaining the difference in outcomes between groups, while remaining undetected. These results highlight the manipulability of SHAP explanations and encourage auditors to treat post-hoc explanations with skepticism.



## **20. Searching for the Essence of Adversarial Perturbations**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15357v1)

**Authors**: Dennis Y. Menn, Hung-yi Lee

**Abstracts**: Neural networks have achieved the state-of-the-art performance on various machine learning fields, yet the incorporation of malicious perturbations with input data (adversarial example) is able to fool neural networks' predictions. This would lead to potential risks in real-world applications, for example, auto piloting and facial recognition. However, the reason for the existence of adversarial examples remains controversial. Here we demonstrate that adversarial perturbations contain human-recognizable information, which is the key conspirator responsible for a neural network's erroneous prediction. This concept of human-recognizable information allows us to explain key features related to adversarial perturbations, which include the existence of adversarial examples, the transferability among different neural networks, and the increased neural network interpretability for adversarial training. Two unique properties in adversarial perturbations that fool neural networks are uncovered: masking and generation. A special class, the complementary class, is identified when neural networks classify input images. The human-recognizable information contained in adversarial perturbations allows researchers to gain insight on the working principles of neural networks and may lead to develop techniques that detect/defense adversarial attacks.



## **21. GAN-based Medical Image Small Region Forgery Detection via a Two-Stage Cascade Framework**

eess.IV

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15170v1)

**Authors**: Jianyi Zhang, Xuanxi Huang, Yaqi Liu, Yuyang Han, Zixiao Xiang

**Abstracts**: Using generative adversarial network (GAN)\cite{RN90} for data enhancement of medical images is significantly helpful for many computer-aided diagnosis (CAD) tasks. A new attack called CT-GAN has emerged. It can inject or remove lung cancer lesions to CT scans. Because the tampering region may even account for less than 1\% of the original image, even state-of-the-art methods are challenging to detect the traces of such tampering.   This paper proposes a cascade framework to detect GAN-based medical image small region forgery like CT-GAN. In the local detection stage, we train the detector network with small sub-images so that interference information in authentic regions will not affect the detector. We use depthwise separable convolution and residual to prevent the detector from over-fitting and enhance the ability to find forged regions through the attention mechanism. The detection results of all sub-images in the same image will be combined into a heatmap. In the global classification stage, using gray level co-occurrence matrix (GLCM) can better extract features of the heatmap. Because the shape and size of the tampered area are uncertain, we train PCA and SVM methods for classification. Our method can classify whether a CT image has been tampered and locate the tampered position. Sufficient experiments show that our method can achieve excellent performance.



## **22. Why Adversarial Training of ReLU Networks Is Difficult?**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15130v1)

**Authors**: Xu Cheng, Hao Zhang, Yue Xin, Wen Shen, Jie Ren, Quanshi Zhang

**Abstracts**: This paper mathematically derives an analytic solution of the adversarial perturbation on a ReLU network, and theoretically explains the difficulty of adversarial training. Specifically, we formulate the dynamics of the adversarial perturbation generated by the multi-step attack, which shows that the adversarial perturbation tends to strengthen eigenvectors corresponding to a few top-ranked eigenvalues of the Hessian matrix of the loss w.r.t. the input. We also prove that adversarial training tends to strengthen the influence of unconfident input samples with large gradient norms in an exponential manner. Besides, we find that adversarial training strengthens the influence of the Hessian matrix of the loss w.r.t. network parameters, which makes the adversarial training more likely to oscillate along directions of a few samples, and boosts the difficulty of adversarial training. Crucially, our proofs provide a unified explanation for previous findings in understanding adversarial training.



## **23. Domain Constraints in Feature Space: Strengthening Robustness of Android Malware Detection against Realizable Adversarial Examples**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15128v1)

**Authors**: Hamid Bostani, Zhuoran Liu, Zhengyu Zhao, Veelasha Moonsamy

**Abstracts**: Strengthening the robustness of machine learning-based malware detectors against realistic evasion attacks remains one of the major obstacles for Android malware detection. To that end, existing work has focused on interpreting domain constraints of Android malware in the problem space, where problem-space realizable adversarial examples are generated. In this paper, we provide another promising way to achieve the same goal but based on interpreting the domain constraints in the feature space, where feature-space realizable adversarial examples are generated. Specifically, we present a novel approach to extracting feature-space domain constraints by learning meaningful feature dependencies from data, and applying them based on a novel robust feature space. Experimental results successfully demonstrate the effectiveness of our novel robust feature space in providing adversarial robustness for DREBIN, a state-of-the-art Android malware detector. For example, it can decrease the evasion rate of a realistic gradient-based attack by $96.4\%$ in a limited-knowledge (transfer) setting and by $13.8\%$ in a more challenging, perfect-knowledge setting. In addition, we show that directly using our learned domain constraints in the adversarial retraining framework leads to about $84\%$ improvement in a limited-knowledge setting, with up to $377\times$ faster implementation than using problem-space adversarial examples.



## **24. Guided Diffusion Model for Adversarial Purification**

cs.CV

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14969v1)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.



## **25. CalFAT: Calibrated Federated Adversarial Training with Label Skewness**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14926v1)

**Authors**: Chen Chen, Yuchen Liu, Xingjun Ma, Lingjuan Lyu

**Abstracts**: Recent studies have shown that, like traditional machine learning, federated learning (FL) is also vulnerable to adversarial attacks. To improve the adversarial robustness of FL, few federated adversarial training (FAT) methods have been proposed to apply adversarial training locally before global aggregation. Although these methods demonstrate promising results on independent identically distributed (IID) data, they suffer from training instability issues on non-IID data with label skewness, resulting in much degraded natural accuracy. This tends to hinder the application of FAT in real-world applications where the label distribution across the clients is often skewed. In this paper, we study the problem of FAT under label skewness, and firstly reveal one root cause of the training instability and natural accuracy degradation issues: skewed labels lead to non-identical class probabilities and heterogeneous local models. We then propose a Calibrated FAT (CalFAT) approach to tackle the instability issue by calibrating the logits adaptively to balance the classes. We show both theoretically and empirically that the optimization of CalFAT leads to homogeneous local models across the clients and much improved convergence rate and final performance.



## **26. CausalAdv: Adversarial Robustness through the Lens of Causality**

cs.LG

ICLR2022, 20 pages, 3 figures

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2106.06196v2)

**Authors**: Yonggang Zhang, Mingming Gong, Tongliang Liu, Gang Niu, Xinmei Tian, Bo Han, Bernhard Schölkopf, Kun Zhang

**Abstracts**: The adversarial vulnerability of deep neural networks has attracted significant attention in machine learning. As causal reasoning has an instinct for modelling distribution change, it is essential to incorporate causality into analyzing this specific type of distribution change induced by adversarial attacks. However, causal formulations of the intuition of adversarial attacks and the development of robust DNNs are still lacking in the literature. To bridge this gap, we construct a causal graph to model the generation process of adversarial examples and define the adversarial distribution to formalize the intuition of adversarial attacks. From the causal perspective, we study the distinction between the natural and adversarial distribution and conclude that the origin of adversarial vulnerability is the focus of models on spurious correlations. Inspired by the causal understanding, we propose the Causal inspired Adversarial distribution alignment method, CausalAdv, to eliminate the difference between natural and adversarial distributions by considering spurious correlations. Extensive experiments demonstrate the efficacy of the proposed method. Our work is the first attempt towards using causality to understand and mitigate the adversarial vulnerability.



## **27. Exposing Fine-grained Adversarial Vulnerability of Face Anti-spoofing Models**

cs.CV

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14851v1)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Bo Peng, Jing Dong

**Abstracts**: Adversarial attacks seriously threaten the high accuracy of face anti-spoofing models. Little adversarial noise can perturb their classification of live and spoofing. The existing adversarial attacks fail to figure out which part of the target face anti-spoofing model is vulnerable, making adversarial analysis tricky. So we propose fine-grained attacks for exposing adversarial vulnerability of face anti-spoofing models. Firstly, we propose Semantic Feature Augmentation (SFA) module, which makes adversarial noise semantic-aware to live and spoofing features. SFA considers the contrastive classes of data and texture bias of models in the context of face anti-spoofing, increasing the attack success rate by nearly 40% on average. Secondly, we generate fine-grained adversarial examples based on SFA and the multitask network with auxiliary information. We evaluate three annotations (facial attributes, spoofing types and illumination) and two geometric maps (depth and reflection), on four backbone networks (VGG, Resnet, Densenet and Swin Transformer). We find that facial attributes annotation and state-of-art networks fail to guarantee that models are robust to adversarial attacks. Such adversarial attacks can be generalized to more auxiliary information and backbone networks, to help our community handle the trade-off between accuracy and adversarial robustness.



## **28. Efficient Reward Poisoning Attacks on Online Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14842v1)

**Authors**: Yinglun Xu, Qi Zeng, Gagandeep Singh

**Abstracts**: We study data poisoning attacks on online deep reinforcement learning (DRL) where the attacker is oblivious to the learning algorithm used by the agent and does not necessarily have full knowledge of the environment. We demonstrate the intrinsic vulnerability of state-of-the-art DRL algorithms by designing a general reward poisoning framework called adversarial MDP attacks. We instantiate our framework to construct several new attacks which only corrupt the rewards for a small fraction of the total training timesteps and make the agent learn a low-performing policy. Our key insight is that the state-of-the-art DRL algorithms strategically explore the environment to find a high-performing policy. Our attacks leverage this insight to construct a corrupted environment for misleading the agent towards learning low-performing policies with a limited attack budget. We provide a theoretical analysis of the efficiency of our attack and perform an extensive evaluation. Our results show that our attacks efficiently poison agents learning with a variety of state-of-the-art DRL algorithms, such as DQN, PPO, SAC, etc. under several popular classical control and MuJoCo environments.



## **29. Mixture GAN For Modulation Classification Resiliency Against Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.15743v1)

**Authors**: Eyad Shtaiwi, Ahmed El Ouadrhiri, Majid Moradikia, Salma Sultana, Ahmed Abdelhadi, Zhu Han

**Abstracts**: Automatic modulation classification (AMC) using the Deep Neural Network (DNN) approach outperforms the traditional classification techniques, even in the presence of challenging wireless channel environments. However, the adversarial attacks cause the loss of accuracy for the DNN-based AMC by injecting a well-designed perturbation to the wireless channels. In this paper, we propose a novel generative adversarial network (GAN)-based countermeasure approach to safeguard the DNN-based AMC systems against adversarial attack examples. GAN-based aims to eliminate the adversarial attack examples before feeding to the DNN-based classifier. Specifically, we have shown the resiliency of our proposed defense GAN against the Fast-Gradient Sign method (FGSM) algorithm as one of the most potent kinds of attack algorithms to craft the perturbed signals. The existing defense-GAN has been designed for image classification and does not work in our case where the above-mentioned communication system is considered. Thus, our proposed countermeasure approach deploys GANs with a mixture of generators to overcome the mode collapsing problem in a typical GAN facing radio signal classification problem. Simulation results show the effectiveness of our proposed defense GAN so that it could enhance the accuracy of the DNN-based AMC under adversarial attacks to 81%, approximately.



## **30. Unfooling Perturbation-Based Post Hoc Explainers**

cs.AI

10 pages (not including references and supplemental)

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14772v1)

**Authors**: Zachariah Carmichael, Walter J Scheirer

**Abstracts**: Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP.



## **31. On the Robustness of Safe Reinforcement Learning under Observational Perturbations**

cs.LG

27 pages, 3 figures, 3 tables

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14691v1)

**Authors**: Zuxin Liu, Zijian Guo, Zhepeng Cen, Huan Zhang, Jie Tan, Bo Li, Ding Zhao

**Abstracts**: Safe reinforcement learning (RL) trains a policy to maximize the task reward while satisfying safety constraints. While prior works focus on the performance optimality, we find that the optimal solutions of many safe RL problems are not robust and safe against carefully designed observational perturbations. We formally analyze the unique properties of designing effective state adversarial attackers in the safe RL setting. We show that baseline adversarial attack techniques for standard RL tasks are not always effective for safe RL and proposed two new approaches - one maximizes the cost and the other maximizes the reward. One interesting and counter-intuitive finding is that the maximum reward attack is strong, as it can both induce unsafe behaviors and make the attack stealthy by maintaining the reward. We further propose a more effective adversarial training framework for safe RL and evaluate it via comprehensive experiments. This work sheds light on the inherited connection between observational robustness and safety in RL and provides a pioneer work for future safe RL studies.



## **32. Superclass Adversarial Attack**

cs.CV

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14629v1)

**Authors**: Soichiro Kumano, Hiroshi Kera, Toshihiko Yamasaki

**Abstracts**: Adversarial attacks have only focused on changing the predictions of the classifier, but their danger greatly depends on how the class is mistaken. For example, when an automatic driving system mistakes a Persian cat for a Siamese cat, it is hardly a problem. However, if it mistakes a cat for a 120km/h minimum speed sign, serious problems can arise. As a stepping stone to more threatening adversarial attacks, we consider the superclass adversarial attack, which causes misclassification of not only fine classes, but also superclasses. We conducted the first comprehensive analysis of superclass adversarial attacks (an existing and 19 new methods) in terms of accuracy, speed, and stability, and identified several strategies to achieve better performance. Although this study is aimed at superclass misclassification, the findings can be applied to other problem settings involving multiple classes, such as top-k and multi-label classification attacks.



## **33. Graph Structure Based Data Augmentation Method**

cs.LG

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14619v1)

**Authors**: Kyung Geun Kim, Byeong Tak Lee

**Abstracts**: In this paper, we propose a novel graph-based data augmentation method that can generally be applied to medical waveform data with graph structures. In the process of recording medical waveform data, such as electrocardiogram (ECG) or electroencephalogram (EEG), angular perturbations between the measurement leads exist due to discrepancies in lead positions. The data samples with large angular perturbations often cause inaccuracy in algorithmic prediction tasks. We design a graph-based data augmentation technique that exploits the inherent graph structures within the medical waveform data to improve both performance and robustness. In addition, we show that the performance gain from graph augmentation results from robustness by testing against adversarial attacks. Since the bases of performance gain are orthogonal, the graph augmentation can be used in conjunction with existing data augmentation techniques to further improve the final performance. We believe that our graph augmentation method opens up new possibilities to explore in data augmentation.



## **34. Problem-Space Evasion Attacks in the Android OS: a Survey**

cs.CR

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14576v1)

**Authors**: Harel Berger, Dr. Chen Hajaj, Dr. Amit Dvir

**Abstracts**: Android is the most popular OS worldwide. Therefore, it is a target for various kinds of malware. As a countermeasure, the security community works day and night to develop appropriate Android malware detection systems, with ML-based or DL-based systems considered as some of the most common types. Against these detection systems, intelligent adversaries develop a wide set of evasion attacks, in which an attacker slightly modifies a malware sample to evade its target detection system. In this survey, we address problem-space evasion attacks in the Android OS, where attackers manipulate actual APKs, rather than their extracted feature vector. We aim to explore this kind of attacks, frequently overlooked by the research community due to a lack of knowledge of the Android domain, or due to focusing on general mathematical evasion attacks - i.e., feature-space evasion attacks. We discuss the different aspects of problem-space evasion attacks, using a new taxonomy, which focuses on key ingredients of each problem-space attack, such as the attacker model, the attacker's mode of operation, and the functional assessment of post-attack applications.



## **35. BadDet: Backdoor Attacks on Object Detection**

cs.CV

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2205.14497v1)

**Authors**: Shih-Han Chan, Yinpeng Dong, Jun Zhu, Xiaolu Zhang, Jun Zhou

**Abstracts**: Deep learning models have been deployed in numerous real-world applications such as autonomous driving and surveillance. However, these models are vulnerable in adversarial environments. Backdoor attack is emerging as a severe security threat which injects a backdoor trigger into a small portion of training data such that the trained model behaves normally on benign inputs but gives incorrect predictions when the specific trigger appears. While most research in backdoor attacks focuses on image classification, backdoor attacks on object detection have not been explored but are of equal importance. Object detection has been adopted as an important module in various security-sensitive applications such as autonomous driving. Therefore, backdoor attacks on object detection could pose severe threats to human lives and properties. We propose four kinds of backdoor attacks for object detection task: 1) Object Generation Attack: a trigger can falsely generate an object of the target class; 2) Regional Misclassification Attack: a trigger can change the prediction of a surrounding object to the target class; 3) Global Misclassification Attack: a single trigger can change the predictions of all objects in an image to the target class; and 4) Object Disappearance Attack: a trigger can make the detector fail to detect the object of the target class. We develop appropriate metrics to evaluate the four backdoor attacks on object detection. We perform experiments using two typical object detection models -- Faster-RCNN and YOLOv3 on different datasets. More crucially, we demonstrate that even fine-tuning on another benign dataset cannot remove the backdoor hidden in the object detection model. To defend against these backdoor attacks, we propose Detector Cleanse, an entropy-based run-time detection framework to identify poisoned testing samples for any deployed object detector.



## **36. Policy Smoothing for Provably Robust Reinforcement Learning**

cs.LG

Published as a conference paper at ICLR 2022

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2106.11420v3)

**Authors**: Aounon Kumar, Alexander Levine, Soheil Feizi

**Abstracts**: The study of provable adversarial robustness for deep neural networks (DNNs) has mainly focused on static supervised learning tasks such as image classification. However, DNNs have been used extensively in real-world adaptive tasks such as reinforcement learning (RL), making such systems vulnerable to adversarial attacks as well. Prior works in provable robustness in RL seek to certify the behaviour of the victim policy at every time-step against a non-adaptive adversary using methods developed for the static setting. But in the real world, an RL adversary can infer the defense strategy used by the victim agent by observing the states, actions, etc., from previous time-steps and adapt itself to produce stronger attacks in future steps. We present an efficient procedure, designed specifically to defend against an adaptive RL adversary, that can directly certify the total reward without requiring the policy to be robust at each time-step. Our main theoretical contribution is to prove an adaptive version of the Neyman-Pearson Lemma -- a key lemma for smoothing-based certificates -- where the adversarial perturbation at a particular time can be a stochastic function of current and previous observations and states as well as previous actions. Building on this result, we propose policy smoothing where the agent adds a Gaussian noise to its observation at each time-step before passing it through the policy function. Our robustness certificates guarantee that the final total reward obtained by policy smoothing remains above a certain threshold, even though the actions at intermediate time-steps may change under the attack. Our experiments on various environments like Cartpole, Pong, Freeway and Mountain Car show that our method can yield meaningful robustness guarantees in practice.



## **37. Certifying Model Accuracy under Distribution Shifts**

cs.LG

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2201.12440v2)

**Authors**: Aounon Kumar, Alexander Levine, Tom Goldstein, Soheil Feizi

**Abstracts**: Certified robustness in machine learning has primarily focused on adversarial perturbations of the input with a fixed attack budget for each point in the data distribution. In this work, we present provable robustness guarantees on the accuracy of a model under bounded Wasserstein shifts of the data distribution. We show that a simple procedure that randomizes the input of the model within a transformation space is provably robust to distributional shifts under the transformation. Our framework allows the datum-specific perturbation size to vary across different points in the input distribution and is general enough to include fixed-sized perturbations as well. Our certificates produce guaranteed lower bounds on the performance of the model for any (natural or adversarial) shift of the input distribution within a Wasserstein ball around the original distribution. We apply our technique to: (i) certify robustness against natural (non-adversarial) transformations of images such as color shifts, hue shifts and changes in brightness and saturation, (ii) certify robustness against adversarial shifts of the input distribution, and (iii) show provable lower bounds (hardness results) on the performance of models trained on so-called "unlearnable" datasets that have been poisoned to interfere with model training.



## **38. SHORTSTACK: Distributed, Fault-tolerant, Oblivious Data Access**

cs.CR

Full version of USENIX OSDI'22 paper

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2205.14281v1)

**Authors**: Midhul Vuppalapati, Kushal Babel, Anurag Khandelwal, Rachit Agarwal

**Abstracts**: Many applications that benefit from data offload to cloud services operate on private data. A now-long line of work has shown that, even when data is offloaded in an encrypted form, an adversary can learn sensitive information by analyzing data access patterns. Existing techniques for oblivious data access--that protect against access pattern attacks--require a centralized, stateful and trusted, proxy to orchestrate data accesses from applications to cloud services. We show that, in failure-prone deployments, such a centralized and stateful proxy results in violation of oblivious data access security guarantees and/or system unavailability. Thus, we initiate the study of distributed, fault-tolerant, oblivious data access.   We present SHORTSTACK, a distributed proxy architecture for oblivious data access in failure-prone deployments. SHORTSTACK achieves the classical obliviousness guarantee--access patterns observed by the adversary being independent of the input--even under a powerful passive persistent adversary that can force failure of arbitrary (bounded-sized) subset of proxy servers at arbitrary times. We also introduce a security model that enables studying oblivious data access with distributed, failure-prone, servers. We provide a formal proof that SHORTSTACK enables oblivious data access under this model, and show empirically that SHORTSTACK performance scales near-linearly with number of distributed proxy servers.



## **39. Semi-supervised Semantics-guided Adversarial Training for Trajectory Prediction**

cs.LG

11 pages, adversarial training for trajectory prediction

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.14230v1)

**Authors**: Ruochen Jiao, Xiangguo Liu, Takami Sato, Qi Alfred Chen, Qi Zhu

**Abstracts**: Predicting the trajectories of surrounding objects is a critical task in self-driving and many other autonomous systems. Recent works demonstrate that adversarial attacks on trajectory prediction, where small crafted perturbations are introduced to history trajectories, may significantly mislead the prediction of future trajectories and ultimately induce unsafe planning. However, few works have addressed enhancing the robustness of this important safety-critical task. In this paper, we present the first adversarial training method for trajectory prediction. Compared with typical adversarial training on image tasks, our work is challenged by more random inputs with rich context, and a lack of class labels. To address these challenges, we propose a method based on a semi-supervised adversarial autoencoder that models disentangled semantic features with domain knowledge and provides additional latent labels for the adversarial training. Extensive experiments with different types of attacks demonstrate that our semi-supervised semantics-guided adversarial training method can effectively mitigate the impact of adversarial attacks and generally improve the system's adversarial robustness to a variety of attacks, including unseen ones. We believe that such semantics-guided architecture and advancement in robust generalization is an important step for developing robust prediction models and enabling safe decision making.



## **40. A Single-Adversary-Single-Detector Zero-Sum Game in Networked Control Systems**

math.OC

6 pages, 6 figures, 1 table, accepted to the 9th IFAC Conference on  Networked Systems, Zurich, July 2022

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.14001v1)

**Authors**: Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev

**Abstracts**: This paper proposes a game-theoretic approach to address the problem of optimal sensor placement for detecting cyber-attacks in networked control systems. The problem is formulated as a zero-sum game with two players, namely a malicious adversary and a detector. Given a protected target vertex, the detector places a sensor at a single vertex to monitor the system and detect the presence of the adversary. On the other hand, the adversary selects a single vertex through which to conduct a cyber-attack that maximally disrupts the target vertex while remaining undetected by the detector. As our first contribution, for a given pair of attack and monitor vertices and a known target vertex, the game payoff function is defined as the output-to-output gain of the respective system. Then, the paper characterizes the set of feasible actions by the detector that ensures bounded values of the game payoff. Finally, an algebraic sufficient condition is proposed to examine whether a given vertex belongs to the set of feasible monitor vertices. The optimal sensor placement is then determined by computing the mixed-strategy Nash equilibrium of the zero-sum game through linear programming. The approach is illustrated via a numerical example of a 10-vertex networked control system with a given target vertex.



## **41. Standalone Neural ODEs with Sensitivity Analysis**

cs.LG

25 pages, 15 figures

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13933v1)

**Authors**: Rym Jaroudi, Lukáš Malý, Gabriel Eilertsen, Tomas B. Johansson, Jonas Unger, George Baravdish

**Abstracts**: This paper presents the Standalone Neural ODE (sNODE), a continuous-depth neural ODE model capable of describing a full deep neural network. This uses a novel nonlinear conjugate gradient (NCG) descent optimization scheme for training, where the Sobolev gradient can be incorporated to improve smoothness of model weights. We also present a general formulation of the neural sensitivity problem and show how it is used in the NCG training. The sensitivity analysis provides a reliable measure of uncertainty propagation throughout a network, and can be used to study model robustness and to generate adversarial attacks. Our evaluations demonstrate that our novel formulations lead to increased robustness and performance as compared to ResNet models, and that it opens up for new opportunities for designing and developing machine learning with improved explainability.



## **42. Evaluating the Robustness of Deep Reinforcement Learning for Autonomous and Adversarial Policies in a Multi-agent Urban Driving Environment**

cs.AI

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2112.11947v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Deep reinforcement learning is actively used for training autonomous and adversarial car policies in a simulated driving environment. Due to the large availability of various reinforcement learning algorithms and the lack of their systematic comparison across different driving scenarios, we are unsure of which ones are more effective for training and testing autonomous car software in single-agent as well as multi-agent driving environments. A benchmarking framework for the comparison of deep reinforcement learning in a vision-based autonomous driving will open up the possibilities for training better autonomous car driving policies. Furthermore, autonomous cars trained on deep reinforcement learning-based algorithms are known for being vulnerable to adversarial attacks. To guard against adversarial attacks, we can train autonomous cars on adversarial driving policies. However, we lack the knowledge of which deep reinforcement learning algorithms would act as good adversarial agents able to effectively test autonomous cars. To address these challenges, we provide an open and reusable benchmarking framework for systematic evaluation and comparative analysis of deep reinforcement learning algorithms for autonomous and adversarial driving in a single- and multi-agent environment. Using the framework, we perform a comparative study of five discrete and two continuous action space deep reinforcement learning algorithms. We run the experiments in a vision-only high fidelity urban driving simulated environments. The results indicate that only some of the deep reinforcement learning algorithms perform consistently better across single and multi-agent scenarios when trained in a multi-agent-only setting.



## **43. Adversarial Deep Reinforcement Learning for Improving the Robustness of Multi-agent Autonomous Driving Policies**

cs.AI

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2112.11937v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Autonomous cars are well known for being vulnerable to adversarial attacks that can compromise the safety of the car and pose danger to other road users. To effectively defend against adversaries, it is required to not only test autonomous cars for finding driving errors, but to improve the robustness of the cars to these errors. To this end, in this paper, we propose a two-step methodology for autonomous cars that consists of (i) finding failure states in autonomous cars by training the adversarial driving agent, and (ii) improving the robustness of autonomous cars by retraining them with effective adversarial inputs. Our methodology supports testing ACs in a multi-agent environment, where we train and compare adversarial car policy on two custom reward functions to test the driving control decision of autonomous cars. We run experiments in a vision-based high fidelity urban driving simulated environment. Our results show that adversarial testing can be used for finding erroneous autonomous driving behavior, followed by adversarial training for improving the robustness of deep reinforcement learning based autonomous driving policies. We demonstrate that the autonomous cars retrained using the effective adversarial inputs noticeably increase the performance of their driving policies in terms of reduced collision and offroad steering errors.



## **44. fakeWeather: Adversarial Attacks for Deep Neural Networks Emulating Weather Conditions on the Camera Lens of Autonomous Systems**

cs.LG

To appear at the 2022 International Joint Conference on Neural  Networks (IJCNN), at the 2022 IEEE World Congress on Computational  Intelligence (WCCI), July 2022, Padua, Italy

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13807v1)

**Authors**: Alberto Marchisio, Giovanni Caramia, Maurizio Martina, Muhammad Shafique

**Abstracts**: Recently, Deep Neural Networks (DNNs) have achieved remarkable performances in many applications, while several studies have enhanced their vulnerabilities to malicious attacks. In this paper, we emulate the effects of natural weather conditions to introduce plausible perturbations that mislead the DNNs. By observing the effects of such atmospheric perturbations on the camera lenses, we model the patterns to create different masks that fake the effects of rain, snow, and hail. Even though the perturbations introduced by our attacks are visible, their presence remains unnoticed due to their association with natural events, which can be especially catastrophic for fully-autonomous and unmanned vehicles. We test our proposed fakeWeather attacks on multiple Convolutional Neural Network and Capsule Network models, and report noticeable accuracy drops in the presence of such adversarial perturbations. Our work introduces a new security threat for DNNs, which is especially severe for safety-critical applications and autonomous systems.



## **45. Face Morphing: Fooling a Face Recognition System Is Simple!**

cs.CV

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13796v1)

**Authors**: Stefan Hörmann, Tianlin Kong, Torben Teepe, Fabian Herzog, Martin Knoche, Gerhard Rigoll

**Abstracts**: State-of-the-art face recognition (FR) approaches have shown remarkable results in predicting whether two faces belong to the same identity, yielding accuracies between 92% and 100% depending on the difficulty of the protocol. However, the accuracy drops substantially when exposed to morphed faces, specifically generated to look similar to two identities. To generate morphed faces, we integrate a simple pretrained FR model into a generative adversarial network (GAN) and modify several loss functions for face morphing. In contrast to previous works, our approach and analyses are not limited to pairs of frontal faces with the same ethnicity and gender. Our qualitative and quantitative results affirm that our approach achieves a seamless change between two faces even in unconstrained scenarios. Despite using features from a simpler FR model for face morphing, we demonstrate that even recent FR systems struggle to distinguish the morphed face from both identities obtaining an accuracy of only 55-70%. Besides, we provide further insights into how knowing the FR system makes it particularly vulnerable to face morphing attacks.



## **46. Adversarial attacks and defenses in Speaker Recognition Systems: A survey**

cs.CR

38pages, 2 figures, 2 tables. Journal of Systems Architecture,2022

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13685v1)

**Authors**: Jiahe Lan, Rui Zhang, Zheng Yan, Jie Wang, Yu Chen, Ronghui Hou

**Abstracts**: Speaker recognition has become very popular in many application scenarios, such as smart homes and smart assistants, due to ease of use for remote control and economic-friendly features. The rapid development of SRSs is inseparable from the advancement of machine learning, especially neural networks. However, previous work has shown that machine learning models are vulnerable to adversarial attacks in the image domain, which inspired researchers to explore adversarial attacks and defenses in Speaker Recognition Systems (SRS). Unfortunately, existing literature lacks a thorough review of this topic. In this paper, we fill this gap by performing a comprehensive survey on adversarial attacks and defenses in SRSs. We first introduce the basics of SRSs and concepts related to adversarial attacks. Then, we propose two sets of criteria to evaluate the performance of attack methods and defense methods in SRSs, respectively. After that, we provide taxonomies of existing attack methods and defense methods, and further review them by employing our proposed criteria. Finally, based on our review, we find some open issues and further specify a number of future directions to motivate the research of SRSs security.



## **47. Sequential Nature of Recommender Systems Disrupts the Evaluation Process**

cs.IR

To Appear in Third International Workshop on Algorithmic Bias in  Search and Recommendation (Bias 2022)

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13681v1)

**Authors**: Ali Shirali

**Abstracts**: Datasets are often generated in a sequential manner, where the previous samples and intermediate decisions or interventions affect subsequent samples. This is especially prominent in cases where there are significant human-AI interactions, such as in recommender systems. To characterize the importance of this relationship across samples, we propose to use adversarial attacks on popular evaluation processes. We present sequence-aware boosting attacks and provide a lower bound on the amount of extra information that can be exploited from a confidential test set solely based on the order of the observed data. We use real and synthetic data to test our methods and show that the evaluation process on the MovieLense-100k dataset can be affected by $\sim1\%$ which is important when considering the close competition. Codes are publicly available.



## **48. Membership Inference Attack Using Self Influence Functions**

cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13680v1)

**Authors**: Gilad Cohen, Raja Giryes

**Abstracts**: Member inference (MI) attacks aim to determine if a specific data sample was used to train a machine learning model. Thus, MI is a major privacy threat to models trained on private sensitive data, such as medical records. In MI attacks one may consider the black-box settings, where the model's parameters and activations are hidden from the adversary, or the white-box case where they are available to the attacker. In this work, we focus on the latter and present a novel MI attack for it that employs influence functions, or more specifically the samples' self-influence scores, to perform the MI prediction. We evaluate our attack on CIFAR-10, CIFAR-100, and Tiny ImageNet datasets, using versatile architectures such as AlexNet, ResNet, and DenseNet. Our attack method achieves new state-of-the-art results for both training with and without data augmentations. Code is available at https://github.com/giladcohen/sif_mi_attack.



## **49. On the Anonymity of Peer-To-Peer Network Anonymity Schemes Used by Cryptocurrencies**

cs.CR

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2201.11860v2)

**Authors**: Piyush Kumar Sharma, Devashish Gosain, Claudia Diaz

**Abstracts**: Cryptocurrency systems can be subject to deanonimization attacks by exploiting the network-level communication on their peer-to-peer network. Adversaries who control a set of colluding node(s) within the peer-to-peer network can observe transactions being exchanged and infer the parties involved. Thus, various network anonymity schemes have been proposed to mitigate this problem, with some solutions providing theoretical anonymity guarantees.   In this work, we model such peer-to-peer network anonymity solutions and evaluate their anonymity guarantees. To do so, we propose a novel framework that uses Bayesian inference to obtain the probability distributions linking transactions to their possible originators. We characterize transaction anonymity with those distributions, using entropy as metric of adversarial uncertainty on the originator's identity. In particular, we model Dandelion, Dandelion++ and Lightning Network. We study different configurations and demonstrate that none of them offers acceptable anonymity to their users. For instance, our analysis reveals that in the widely deployed Lightning Network, with 1% strategically chosen colluding nodes the adversary can uniquely determine the originator for about 50% of the total transactions in the network. In Dandelion, an adversary that controls 15% of the nodes has on average uncertainty among only 8 possible originators. Moreover, we observe that due to the way Dandelion and Dandelion++ are designed, increasing the network size does not correspond to an increase in the anonymity set of potential originators. Alarmingly, our longitudinal analysis of Lightning Network reveals rather an inverse trend -- with the growth of the network the overall anonymity decreases.



## **50. Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks**

cs.CR

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2111.12965v2)

**Authors**: Xiangyu Qi, Tinghao Xie, Ruizhe Pan, Jifeng Zhu, Yong Yang, Kai Bu

**Abstracts**: One major goal of the AI security community is to securely and reliably produce and deploy deep learning models for real-world applications. To this end, data poisoning based backdoor attacks on deep neural networks (DNNs) in the production stage (or training stage) and corresponding defenses are extensively explored in recent years. Ironically, backdoor attacks in the deployment stage, which can often happen in unprofessional users' devices and are thus arguably far more threatening in real-world scenarios, draw much less attention of the community. We attribute this imbalance of vigilance to the weak practicality of existing deployment-stage backdoor attack algorithms and the insufficiency of real-world attack demonstrations. To fill the blank, in this work, we study the realistic threat of deployment-stage backdoor attacks on DNNs. We base our study on a commonly used deployment-stage attack paradigm -- adversarial weight attack, where adversaries selectively modify model weights to embed backdoor into deployed DNNs. To approach realistic practicality, we propose the first gray-box and physically realizable weights attack algorithm for backdoor injection, namely subnet replacement attack (SRA), which only requires architecture information of the victim model and can support physical triggers in the real world. Extensive experimental simulations and system-level real-world attack demonstrations are conducted. Our results not only suggest the effectiveness and practicality of the proposed attack algorithm, but also reveal the practical risk of a novel type of computer virus that may widely spread and stealthily inject backdoor into DNN models in user devices. By our study, we call for more attention to the vulnerability of DNNs in the deployment stage.



