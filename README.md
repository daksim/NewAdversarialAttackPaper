# Latest Adversarial Attack Papers
**update at 2022-06-07 06:31:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Gradient Obfuscation Checklist Test Gives a False Sense of Security**

cs.CV

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01705v1)

**Authors**: Nikola Popovic, Danda Pani Paudel, Thomas Probst, Luc Van Gool

**Abstracts**: One popular group of defense techniques against adversarial attacks is based on injecting stochastic noise into the network. The main source of robustness of such stochastic defenses however is often due to the obfuscation of the gradients, offering a false sense of security. Since most of the popular adversarial attacks are optimization-based, obfuscated gradients reduce their attacking ability, while the model is still susceptible to stronger or specifically tailored adversarial attacks. Recently, five characteristics have been identified, which are commonly observed when the improvement in robustness is mainly caused by gradient obfuscation. It has since become a trend to use these five characteristics as a sufficient test, to determine whether or not gradient obfuscation is the main source of robustness. However, these characteristics do not perfectly characterize all existing cases of gradient obfuscation, and therefore can not serve as a basis for a conclusive test. In this work, we present a counterexample, showing this test is not sufficient for concluding that gradient obfuscation is not the main cause of improvements in robustness.



## **2. Evaluating Transfer-based Targeted Adversarial Perturbations against Real-World Computer Vision Systems based on Human Judgments**

cs.CV

technical report

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01467v1)

**Authors**: Zhengyu Zhao, Nga Dang, Martha Larson

**Abstracts**: Computer vision systems are remarkably vulnerable to adversarial perturbations. Transfer-based adversarial images are generated on one (source) system and used to attack another (target) system. In this paper, we take the first step to investigate transfer-based targeted adversarial images in a realistic scenario where the target system is trained on some private data with its inventory of semantic labels not publicly available. Our main contributions include an extensive human-judgment-based evaluation of attack success on the Google Cloud Vision API and additional analysis of the different behaviors of Google Cloud Vision in face of original images vs. adversarial images. Resources are publicly available at \url{https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/google_results.zip}.



## **3. Adversarial Attacks on Human Vision**

cs.CV

21 pages, 8 figures, 1 table

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01365v1)

**Authors**: Victor A. Mateescu, Ivan V. Bajić

**Abstracts**: This article presents an introduction to visual attention retargeting, its connection to visual saliency, the challenges associated with it, and ideas for how it can be approached. The difficulty of attention retargeting as a saliency inversion problem lies in the lack of one-to-one mapping between saliency and the image domain, in addition to the possible negative impact of saliency alterations on image aesthetics. A few approaches from recent literature to solve this challenging problem are reviewed, and several suggestions for future development are presented.



## **4. On the Privacy Properties of GAN-generated Samples**

cs.LG

AISTATS 2021

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01349v1)

**Authors**: Zinan Lin, Vyas Sekar, Giulia Fanti

**Abstracts**: The privacy implications of generative adversarial networks (GANs) are a topic of great interest, leading to several recent algorithms for training GANs with privacy guarantees. By drawing connections to the generalization properties of GANs, we prove that under some assumptions, GAN-generated samples inherently satisfy some (weak) privacy guarantees. First, we show that if a GAN is trained on m samples and used to generate n samples, the generated samples are (epsilon, delta)-differentially-private for (epsilon, delta) pairs where delta scales as O(n/m). We show that under some special conditions, this upper bound is tight. Next, we study the robustness of GAN-generated samples to membership inference attacks. We model membership inference as a hypothesis test in which the adversary must determine whether a given sample was drawn from the training dataset or from the underlying data distribution. We show that this adversary can achieve an area under the ROC curve that scales no better than O(m^{-1/4}).



## **5. A Barrier Certificate-based Simplex Architecture with Application to Microgrids**

eess.SY

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2202.09710v2)

**Authors**: Amol Damare, Shouvik Roy, Scott A. Smolka, Scott D. Stoller

**Abstracts**: We present Barrier Certificate-based Simplex (BC-Simplex), a new, provably correct design for runtime assurance of continuous dynamical systems. BC-Simplex is centered around the Simplex Control Architecture, which consists of a high-performance advanced controller which is not guaranteed to maintain safety of the plant, a verified-safe baseline controller, and a decision module that switches control of the plant between the two controllers to ensure safety without sacrificing performance. In BC-Simplex, Barrier certificates are used to prove that the baseline controller ensures safety. Furthermore, BC-Simplex features a new automated method for deriving, from the barrier certificate, the conditions for switching between the controllers. Our method is based on the Taylor expansion of the barrier certificate and yields computationally inexpensive switching conditions. We consider a significant application of BC-Simplex to a microgrid featuring an advanced controller in the form of a neural network trained using reinforcement learning. The microgrid is modeled in RTDS, an industry-standard high-fidelity, real-time power systems simulator. Our results demonstrate that BC-Simplex can automatically derive switching conditions for complex systems, the switching conditions are not overly conservative, and BC-Simplex ensures safety even in the presence of adversarial attacks on the neural controller.



## **6. Adversarial Laser Spot: Robust and Covert Physical Adversarial Attack to DNNs**

cs.CV

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.01034v1)

**Authors**: Chengyin Hu

**Abstracts**: Most existing deep neural networks (DNNs) are easily disturbed by slight noise. As far as we know, there are few researches on physical adversarial attack technology by deploying lighting equipment. The light-based physical adversarial attack technology has excellent covertness, which brings great security risks to many applications based on deep neural networks (such as automatic driving technology). Therefore, we propose a robust physical adversarial attack technology with excellent covertness, called adversarial laser point (AdvLS), which optimizes the physical parameters of laser point through genetic algorithm to perform physical adversarial attack. It realizes robust and covert physical adversarial attack by using low-cost laser equipment. As far as we know, AdvLS is the first light-based adversarial attack technology that can perform physical adversarial attacks in the daytime. A large number of experiments in the digital and physical environments show that AdvLS has excellent robustness and concealment. In addition, through in-depth analysis of the experimental data, we find that the adversarial perturbations generated by AdvLS have superior adversarial attack migration. The experimental results show that AdvLS impose serious interference to the advanced deep neural networks, we call for the attention of the proposed physical adversarial attack technology.



## **7. FACM: Correct the Output of Deep Neural Network with Middle Layers Features against Adversarial Samples**

cs.CV

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.00924v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: In the strong adversarial attacks against deep neural network (DNN), the output of DNN will be misclassified if and only if the last feature layer of the DNN is completely destroyed by adversarial samples, while our studies found that the middle feature layers of the DNN can still extract the effective features of the original normal category in these adversarial attacks. To this end, in this paper, a middle $\bold{F}$eature layer $\bold{A}$nalysis and $\bold{C}$onditional $\bold{M}$atching prediction distribution (FACM) model is proposed to increase the robustness of the DNN against adversarial samples through correcting the output of DNN with the features extracted by the middle layers of DNN. In particular, the middle $\bold{F}$eature layer $\bold{A}$nalysis (FA) module, the conditional matching prediction distribution (CMPD) module and the output decision module are included in our FACM model to collaboratively correct the classification of adversarial samples. The experiments results show that, our FACM model can significantly improve the robustness of the naturally trained model against various attacks, and our FA model can significantly improve the robustness of the adversarially trained model against white-box attacks with weak transferability and black box attacks where FA model includes the FA module and the output decision module, not the CMPD module.



## **8. Mask-Guided Divergence Loss Improves the Generalization and Robustness of Deep Neural Network**

cs.LG

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.00913v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Deep neural network (DNN) with dropout can be regarded as an ensemble model consisting of lots of sub-DNNs (i.e., an ensemble sub-DNN where the sub-DNN is the remaining part of the DNN after dropout), and through increasing the diversity of the ensemble sub-DNN, the generalization and robustness of the DNN can be effectively improved. In this paper, a mask-guided divergence loss function (MDL), which consists of a cross-entropy loss term and an orthogonal term, is proposed to increase the diversity of the ensemble sub-DNN by the added orthogonal term. Particularly, the mask technique is introduced to assist in generating the orthogonal term for avoiding overfitting of the diversity learning. The theoretical analysis and extensive experiments on 4 datasets (i.e., MNIST, FashionMNIST, CIFAR10, and CIFAR100) manifest that MDL can improve the generalization and robustness of standard training and adversarial training. For CIFAR10 and CIFAR100, in standard training, the maximum improvement of accuracy is $1.38\%$ on natural data, $30.97\%$ on FGSM (i.e., Fast Gradient Sign Method) attack, $38.18\%$ on PGD (i.e., Projected Gradient Descent) attack. While in adversarial training, the maximum improvement is $1.68\%$ on natural data, $4.03\%$ on FGSM attack and $2.65\%$ on PGD attack.



## **9. Robust Feature-Level Adversaries are Interpretability Tools**

cs.LG

Code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2110.03605v4)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstracts**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying the representations in models. Second, we show that these adversaries are versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results indicate that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations.



## **10. On the reversibility of adversarial attacks**

cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00772v1)

**Authors**: Chau Yi Li, Ricardo Sánchez-Matilla, Ali Shahin Shamsabadi, Riccardo Mazzon, Andrea Cavallaro

**Abstracts**: Adversarial attacks modify images with perturbations that change the prediction of classifiers. These modified images, known as adversarial examples, expose the vulnerabilities of deep neural network classifiers. In this paper, we investigate the predictability of the mapping between the classes predicted for original images and for their corresponding adversarial examples. This predictability relates to the possibility of retrieving the original predictions and hence reversing the induced misclassification. We refer to this property as the reversibility of an adversarial attack, and quantify reversibility as the accuracy in retrieving the original class or the true class of an adversarial example. We present an approach that reverses the effect of an adversarial attack on a classifier using a prior set of classification results. We analyse the reversibility of state-of-the-art adversarial attacks on benchmark classifiers and discuss the factors that affect the reversibility.



## **11. Training privacy-preserving video analytics pipelines by suppressing features that reveal information about private attributes**

cs.CV

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2203.02635v2)

**Authors**: Chau Yi Li, Andrea Cavallaro

**Abstracts**: Deep neural networks are increasingly deployed for scene analytics, including to evaluate the attention and reaction of people exposed to out-of-home advertisements. However, the features extracted by a deep neural network that was trained to predict a specific, consensual attribute (e.g. emotion) may also encode and thus reveal information about private, protected attributes (e.g. age or gender). In this work, we focus on such leakage of private information at inference time. We consider an adversary with access to the features extracted by the layers of a deployed neural network and use these features to predict private attributes. To prevent the success of such an attack, we modify the training of the network using a confusion loss that encourages the extraction of features that make it difficult for the adversary to accurately predict private attributes. We validate this training approach on image-based tasks using a publicly available dataset. Results show that, compared to the original network, the proposed PrivateNet can reduce the leakage of private information of a state-of-the-art emotion recognition classifier by 2.88% for gender and by 13.06% for age group, with a minimal effect on task accuracy.



## **12. Adversarial Attacks on Gaussian Process Bandits**

stat.ML

Accepted to ICML 2022

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2110.08449v2)

**Authors**: Eric Han, Jonathan Scarlett

**Abstracts**: Gaussian processes (GP) are a widely-adopted tool used to sequentially optimize black-box functions, where evaluations are costly and potentially noisy. Recent works on GP bandits have proposed to move beyond random noise and devise algorithms robust to adversarial attacks. This paper studies this problem from the attacker's perspective, proposing various adversarial attack methods with differing assumptions on the attacker's strength and prior information. Our goal is to understand adversarial attacks on GP bandits from theoretical and practical perspectives. We focus primarily on targeted attacks on the popular GP-UCB algorithm and a related elimination-based algorithm, based on adversarially perturbing the function $f$ to produce another function $\tilde{f}$ whose optima are in some target region $\mathcal{R}_{\rm target}$. Based on our theoretical analysis, we devise both white-box attacks (known $f$) and black-box attacks (unknown $f$), with the former including a Subtraction attack and Clipping attack, and the latter including an Aggressive subtraction attack. We demonstrate that adversarial attacks on GP bandits can succeed in forcing the algorithm towards $\mathcal{R}_{\rm target}$ even with a low attack budget, and we test our attacks' effectiveness on a diverse range of objective functions.



## **13. The robust way to stack and bag: the local Lipschitz way**

cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00513v1)

**Authors**: Thulasi Tholeti, Sheetal Kalyani

**Abstracts**: Recent research has established that the local Lipschitz constant of a neural network directly influences its adversarial robustness. We exploit this relationship to construct an ensemble of neural networks which not only improves the accuracy, but also provides increased adversarial robustness. The local Lipschitz constants for two different ensemble methods - bagging and stacking - are derived and the architectures best suited for ensuring adversarial robustness are deduced. The proposed ensemble architectures are tested on MNIST and CIFAR-10 datasets in the presence of white-box attacks, FGSM and PGD. The proposed architecture is found to be more robust than a) a single network and b) traditional ensemble methods.



## **14. Attack-Agnostic Adversarial Detection**

cs.CV

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00489v1)

**Authors**: Jiaxin Cheng, Mohamed Hussein, Jay Billa, Wael AbdAlmageed

**Abstracts**: The growing number of adversarial attacks in recent years gives attackers an advantage over defenders, as defenders must train detectors after knowing the types of attacks, and many models need to be maintained to ensure good performance in detecting any upcoming attacks. We propose a way to end the tug-of-war between attackers and defenders by treating adversarial attack detection as an anomaly detection problem so that the detector is agnostic to the attack. We quantify the statistical deviation caused by adversarial perturbations in two aspects. The Least Significant Component Feature (LSCF) quantifies the deviation of adversarial examples from the statistics of benign samples and Hessian Feature (HF) reflects how adversarial examples distort the landscape of the model's optima by measuring the local loss curvature. Empirical results show that our method can achieve an overall ROC AUC of 94.9%, 89.7%, and 94.6% on CIFAR10, CIFAR100, and SVHN, respectively, and has comparable performance to adversarial detectors trained with adversarial examples on most of the attacks.



## **15. Generating End-to-End Adversarial Examples for Malware Classifiers Using Explainability**

cs.CR

Accepted as a conference paper at IJCNN 2020

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2009.13243v2)

**Authors**: Ishai Rosenberg, Shai Meir, Jonathan Berrebi, Ilay Gordon, Guillaume Sicard, Eli David

**Abstracts**: In recent years, the topic of explainable machine learning (ML) has been extensively researched. Up until now, this research focused on regular ML users use-cases such as debugging a ML model. This paper takes a different posture and show that adversaries can leverage explainable ML to bypass multi-feature types malware classifiers. Previous adversarial attacks against such classifiers only add new features and not modify existing ones to avoid harming the modified malware executable's functionality. Current attacks use a single algorithm that both selects which features to modify and modifies them blindly, treating all features the same. In this paper, we present a different approach. We split the adversarial example generation task into two parts: First we find the importance of all features for a specific sample using explainability algorithms, and then we conduct a feature-specific modification, feature-by-feature. In order to apply our attack in black-box scenarios, we introduce the concept of transferability of explainability, that is, applying explainability algorithms to different classifiers using different features subsets and trained on different datasets still result in a similar subset of important features. We conclude that explainability algorithms can be leveraged by adversaries and thus the advocates of training more interpretable classifiers should consider the trade-off of higher vulnerability of those classifiers to adversarial attacks.



## **16. Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations**

cs.CR

Accepted by IJCAI 2022

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00477v1)

**Authors**: Run Wang, Ziheng Huang, Zhikai Chen, Li Liu, Jing Chen, Lina Wang

**Abstracts**: DeepFake is becoming a real risk to society and brings potential threats to both individual privacy and political security due to the DeepFaked multimedia are realistic and convincing. However, the popular DeepFake passive detection is an ex-post forensics countermeasure and failed in blocking the disinformation spreading in advance. To address this limitation, researchers study the proactive defense techniques by adding adversarial noises into the source data to disrupt the DeepFake manipulation. However, the existing studies on proactive DeepFake defense via injecting adversarial noises are not robust, which could be easily bypassed by employing simple image reconstruction revealed in a recent study MagDR.   In this paper, we investigate the vulnerability of the existing forgery techniques and propose a novel \emph{anti-forgery} technique that helps users protect the shared facial images from attackers who are capable of applying the popular forgery techniques. Our proposed method generates perceptual-aware perturbations in an incessant manner which is vastly different from the prior studies by adding adversarial noises that is sparse. Experimental results reveal that our perceptual-aware perturbations are robust to diverse image transformations, especially the competitive evasion technique, MagDR via image reconstruction. Our findings potentially open up a new research direction towards thorough understanding and investigation of perceptual-aware adversarial attack for protecting facial images against DeepFakes in a proactive and robust manner. We open-source our tool to foster future research. Code is available at https://github.com/AbstractTeen/AntiForgery/.



## **17. PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations**

cs.CR

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2205.13523v2)

**Authors**: Manaar Alam, Esha Sarkar, Michail Maniatakos

**Abstracts**: Federated Learning (FL) enables numerous participants to train deep learning models collaboratively without exposing their personal, potentially sensitive data, making it a promising solution for data privacy in collaborative training. The distributed nature of FL and unvetted data, however, makes it inherently vulnerable to backdoor attacks: In this scenario, an adversary injects backdoor functionality into the centralized model during training, which can be triggered to cause the desired misclassification for a specific adversary-chosen input. A range of prior work establishes successful backdoor injection in an FL system; however, these backdoors are not demonstrated to be long-lasting. The backdoor functionality does not remain in the system if the adversary is removed from the training process since the centralized model parameters continuously mutate during successive FL training rounds. Therefore, in this work, we propose PerDoor, a persistent-by-construction backdoor injection technique for FL, driven by adversarial perturbation and targeting parameters of the centralized model that deviate less in successive FL rounds and contribute the least to the main task accuracy. An exhaustive evaluation considering an image classification scenario portrays on average $10.5\times$ persistence over multiple FL rounds compared to traditional backdoor attacks. Through experiments, we further exhibit the potency of PerDoor in the presence of state-of-the-art backdoor prevention techniques in an FL system. Additionally, the operation of adversarial perturbation also assists PerDoor in developing non-uniform trigger patterns for backdoor inputs compared to uniform triggers (with fixed patterns and locations) of existing backdoor techniques, which are prone to be easily mitigated.



## **18. NeuroUnlock: Unlocking the Architecture of Obfuscated Deep Neural Networks**

cs.CR

The definitive Version of Record will be Published in the 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00402v1)

**Authors**: Mahya Morid Ahmadi, Lilas Alrahis, Alessio Colucci, Ozgur Sinanoglu, Muhammad Shafique

**Abstracts**: The advancements of deep neural networks (DNNs) have led to their deployment in diverse settings, including safety and security-critical applications. As a result, the characteristics of these models have become sensitive intellectual properties that require protection from malicious users. Extracting the architecture of a DNN through leaky side-channels (e.g., memory access) allows adversaries to (i) clone the model, and (ii) craft adversarial attacks. DNN obfuscation thwarts side-channel-based architecture stealing (SCAS) attacks by altering the run-time traces of a given DNN while preserving its functionality. In this work, we expose the vulnerability of state-of-the-art DNN obfuscation methods to these attacks. We present NeuroUnlock, a novel SCAS attack against obfuscated DNNs. Our NeuroUnlock employs a sequence-to-sequence model that learns the obfuscation procedure and automatically reverts it, thereby recovering the original DNN architecture. We demonstrate the effectiveness of NeuroUnlock by recovering the architecture of 200 randomly generated and obfuscated DNNs running on the Nvidia RTX 2080 TI graphics processing unit (GPU). Moreover, NeuroUnlock recovers the architecture of various other obfuscated DNNs, such as the VGG-11, VGG-13, ResNet-20, and ResNet-32 networks. After recovering the architecture, NeuroUnlock automatically builds a near-equivalent DNN with only a 1.4% drop in the testing accuracy. We further show that launching a subsequent adversarial attack on the recovered DNNs boosts the success rate of the adversarial attack by 51.7% in average compared to launching it on the obfuscated versions. Additionally, we propose a novel methodology for DNN obfuscation, ReDLock, which eradicates the deterministic nature of the obfuscation and achieves 2.16X more resilience to the NeuroUnlock attack. We release the NeuroUnlock and the ReDLock as open-source frameworks.



## **19. Support Vector Machines under Adversarial Label Contamination**

cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00352v1)

**Authors**: Huang Xiao, Battista Biggio, Blaine Nelson, Han Xiao, Claudia Eckert, Fabio Roli

**Abstracts**: Machine learning algorithms are increasingly being applied in security-related tasks such as spam and malware detection, although their security properties against deliberate attacks have not yet been widely understood. Intelligent and adaptive attackers may indeed exploit specific vulnerabilities exposed by machine learning techniques to violate system security. Being robust to adversarial data manipulation is thus an important, additional requirement for machine learning algorithms to successfully operate in adversarial settings. In this work, we evaluate the security of Support Vector Machines (SVMs) to well-crafted, adversarial label noise attacks. In particular, we consider an attacker that aims to maximize the SVM's classification error by flipping a number of labels in the training data. We formalize a corresponding optimal attack strategy, and solve it by means of heuristic approaches to keep the computational complexity tractable. We report an extensive experimental analysis on the effectiveness of the considered attacks against linear and non-linear SVMs, both on synthetic and real-world datasets. We finally argue that our approach can also provide useful insights for developing more secure SVM learning algorithms, and also novel techniques in a number of related research areas, such as semi-supervised and active learning.



## **20. A Simple Structure For Building A Robust Model**

cs.CV

Accepted by Fifth International Conference on Intelligence Science  (ICIS2022); 10 pages, 3 figures, 4 tables

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2204.11596v2)

**Authors**: Xiao Tan, Jingbo Gao, Ruolin Li

**Abstracts**: As deep learning applications, especially programs of computer vision, are increasingly deployed in our lives, we have to think more urgently about the security of these applications.One effective way to improve the security of deep learning models is to perform adversarial training, which allows the model to be compatible with samples that are deliberately created for use in attacking the model.Based on this, we propose a simple architecture to build a model with a certain degree of robustness, which improves the robustness of the trained network by adding an adversarial sample detection network for cooperative training. At the same time, we design a new data sampling strategy that incorporates multiple existing attacks, allowing the model to adapt to many different adversarial attacks with a single training.We conducted some experiments to test the effectiveness of this design based on Cifar10 dataset, and the results indicate that it has some degree of positive effect on the robustness of the model.Our code could be found at https://github.com/dowdyboy/simple_structure_for_robust_model .



## **21. Bounding Membership Inference**

cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2202.12232v2)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $\epsilon$-DP or $(\epsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.



## **22. Metamorphic Testing-based Adversarial Attack to Fool Deepfake Detectors**

cs.CV

paper accepted at 26TH International Conference on Pattern  Recognition (ICPR2022)

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2204.08612v2)

**Authors**: Nyee Thoang Lim, Meng Yi Kuan, Muxin Pu, Mei Kuan Lim, Chun Yong Chong

**Abstracts**: Deepfakes utilise Artificial Intelligence (AI) techniques to create synthetic media where the likeness of one person is replaced with another. There are growing concerns that deepfakes can be maliciously used to create misleading and harmful digital contents. As deepfakes become more common, there is a dire need for deepfake detection technology to help spot deepfake media. Present deepfake detection models are able to achieve outstanding accuracy (>90%). However, most of them are limited to within-dataset scenario, where the same dataset is used for training and testing. Most models do not generalise well enough in cross-dataset scenario, where models are tested on unseen datasets from another source. Furthermore, state-of-the-art deepfake detection models rely on neural network-based classification models that are known to be vulnerable to adversarial attacks. Motivated by the need for a robust deepfake detection model, this study adapts metamorphic testing (MT) principles to help identify potential factors that could influence the robustness of the examined model, while overcoming the test oracle problem in this domain. Metamorphic testing is specifically chosen as the testing technique as it fits our demand to address learning-based system testing with probabilistic outcomes from largely black-box components, based on potentially large input domains. We performed our evaluations on MesoInception-4 and TwoStreamNet models, which are the state-of-the-art deepfake detection models. This study identified makeup application as an adversarial attack that could fool deepfake detectors. Our experimental results demonstrate that both the MesoInception-4 and TwoStreamNet models degrade in their performance by up to 30\% when the input data is perturbed with makeup.



## **23. FoveaTer: Foveated Transformer for Image Classification**

cs.CV

17 pages, 7 figures

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2105.14173v2)

**Authors**: Aditya Jonnalagadda, William Yang Wang, B. S. Manjunath, Miguel P. Eckstein

**Abstracts**: Many animals and humans process the visual field with a varying spatial resolution (foveated vision) and use peripheral processing to make eye movements and point the fovea to acquire high-resolution information about objects of interest. This architecture results in computationally efficient rapid scene exploration. Recent progress in self-attention-based vision Transformers allow global interactions between feature locations and result in increased robustness to adversarial attacks. However, the Transformer models do not explicitly model the foveated properties of the visual system nor the interaction between eye movements and the classification task. We propose foveated Transformer (FoveaTer) model, which uses pooling regions and eye movements to perform object classification tasks. Our proposed model pools the image features using squared pooling regions, an approximation to the biologically-inspired foveated architecture. It decides on subsequent fixation locations based on the attention assigned by the Transformer to various locations from past and present fixations. It dynamically allocates more fixation/computational resources to more challenging images before making the final object category decision. We compare FoveaTer against a Full-resolution baseline model, which does not contain any pooling. On the ImageNet dataset, the Foveated model with Dynamic-stop achieves an accuracy of $1.9\%$ below the full-resolution model with a throughput gain of $51\%$. Using a Foveated model with Dynamic-stop and the Full-resolution model, the ensemble outperforms the baseline Full-resolution by $0.2\%$ with a throughput gain of $7.7\%$. We also demonstrate our model's robustness against adversarial attacks. Finally, we compare the Foveated model to human performance in a scene categorization task and show similar dependence of accuracy with number of exploratory fixations.



## **24. Generative Models with Information-Theoretic Protection Against Membership Inference Attacks**

cs.LG

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2206.00071v1)

**Authors**: Parisa Hassanzadeh, Robert E. Tillman

**Abstracts**: Deep generative models, such as Generative Adversarial Networks (GANs), synthesize diverse high-fidelity data samples by estimating the underlying distribution of high dimensional data. Despite their success, GANs may disclose private information from the data they are trained on, making them susceptible to adversarial attacks such as membership inference attacks, in which an adversary aims to determine if a record was part of the training set. We propose an information theoretically motivated regularization term that prevents the generative model from overfitting to training data and encourages generalizability. We show that this penalty minimizes the JensenShannon divergence between components of the generator trained on data with different membership, and that it can be implemented at low cost using an additional classifier. Our experiments on image datasets demonstrate that with the proposed regularization, which comes at only a small added computational cost, GANs are able to preserve privacy and generate high-quality samples that achieve better downstream classification performance compared to non-private and differentially private generative models.



## **25. CodeAttack: Code-based Adversarial Attacks for Pre-Trained Programming Language Models**

cs.CL

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2206.00052v1)

**Authors**: Akshita Jha, Chandan K. Reddy

**Abstracts**: Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks. We propose, CodeAttack, a simple yet effective black-box attack model that uses code structure to generate imperceptible, effective, and minimally perturbed adversarial code samples. We demonstrate the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall performance while being more efficient and imperceptible.



## **26. Hide and Seek: on the Stealthiness of Attacks against Deep Learning Systems**

cs.CR

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2205.15944v1)

**Authors**: Zeyan Liu, Fengjun Li, Jingqiang Lin, Zhu Li, Bo Luo

**Abstracts**: With the growing popularity of artificial intelligence and machine learning, a wide spectrum of attacks against deep learning models have been proposed in the literature. Both the evasion attacks and the poisoning attacks attempt to utilize adversarially altered samples to fool the victim model to misclassify the adversarial sample. While such attacks claim to be or are expected to be stealthy, i.e., imperceptible to human eyes, such claims are rarely evaluated. In this paper, we present the first large-scale study on the stealthiness of adversarial samples used in the attacks against deep learning. We have implemented 20 representative adversarial ML attacks on six popular benchmarking datasets. We evaluate the stealthiness of the attack samples using two complementary approaches: (1) a numerical study that adopts 24 metrics for image similarity or quality assessment; and (2) a user study of 3 sets of questionnaires that has collected 20,000+ annotations from 1,000+ responses. Our results show that the majority of the existing attacks introduce nonnegligible perturbations that are not stealthy to human eyes. We further analyze the factors that contribute to attack stealthiness. We further examine the correlation between the numerical analysis and the user studies, and demonstrate that some image quality metrics may provide useful guidance in attack designs, while there is still a significant gap between assessed image quality and visual stealthiness of attacks.



## **27. Atomic cross-chain exchanges of shared assets**

cs.CR

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2202.12855v2)

**Authors**: Krishnasuri Narayanam, Venkatraman Ramakrishna, Dhinakaran Vinayagamurthy, Sandeep Nishad

**Abstracts**: A core enabler for blockchain or DLT interoperability is the ability to atomically exchange assets held by mutually untrusting owners on different ledgers. This atomic swap problem has been well-studied, with the Hash Time Locked Contract (HTLC) emerging as a canonical solution. HTLC ensures atomicity of exchange, albeit with caveats for node failure and timeliness of claims. But a bigger limitation of HTLC is that it only applies to a model consisting of two adversarial parties having sole ownership of a single asset in each ledger. Realistic extensions of the model in which assets may be jointly owned by multiple parties, all of whose consents are required for exchanges, or where multiple assets must be exchanged for one, are susceptible to collusion attacks and hence cannot be handled by HTLC. In this paper, we generalize the model of asset exchanges across DLT networks and present a taxonomy of use cases, describe the threat model, and propose MPHTLC, an augmented HTLC protocol for atomic multi-owner-and-asset exchanges. We analyze the correctness, safety, and application scope of MPHTLC. As proof-of-concept, we show how MPHTLC primitives can be implemented in networks built on Hyperledger Fabric and Corda, and how MPHTLC can be implemented in the Hyperledger Labs Weaver framework by augmenting its existing HTLC protocol.



## **28. Semantic Autoencoder and Its Potential Usage for Adversarial Attack**

cs.LG

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2205.15592v1)

**Authors**: Yurui Ming, Cuihuan Du, Chin-Teng Lin

**Abstracts**: Autoencoder can give rise to an appropriate latent representation of the input data, however, the representation which is solely based on the intrinsic property of the input data, is usually inferior to express some semantic information. A typical case is the potential incapability of forming a clear boundary upon clustering of these representations. By encoding the latent representation that not only depends on the content of the input data, but also the semantic of the input data, such as label information, we propose an enhanced autoencoder architecture named semantic autoencoder. Experiments of representation distribution via t-SNE shows a clear distinction between these two types of encoders and confirm the supremacy of the semantic one, whilst the decoded samples of these two types of autoencoders exhibit faint dissimilarity either objectively or subjectively. Based on this observation, we consider adversarial attacks to learning algorithms that rely on the latent representation obtained via autoencoders. It turns out that latent contents of adversarial samples constructed from semantic encoder with deliberate wrong label information exhibit different distribution compared with that of the original input data, while both of these samples manifest very marginal difference. This new way of attack set up by our work is worthy of attention due to the necessity to secure the widespread deep learning applications.



## **29. Connecting adversarial attacks and optimal transport for domain adaptation**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15424v1)

**Authors**: Arip Asadulaev, Vitaly Shutov, Alexander Korotin, Alexander Panfilov, Andrey Filchenkov

**Abstracts**: We present a novel algorithm for domain adaptation using optimal transport. In domain adaptation, the goal is to adapt a classifier trained on the source domain samples to the target domain. In our method, we use optimal transport to map target samples to the domain named source fiction. This domain differs from the source but is accurately classified by the source domain classifier. Our main idea is to generate a source fiction by c-cyclically monotone transformation over the target domain. If samples with the same labels in two domains are c-cyclically monotone, the optimal transport map between these domains preserves the class-wise structure, which is the main goal of domain adaptation. To generate a source fiction domain, we propose an algorithm that is based on our finding that adversarial attacks are a c-cyclically monotone transformation of the dataset. We conduct experiments on Digits and Modern Office-31 datasets and achieve improvement in performance for simple discrete optimal transport solvers for all adaptation tasks.



## **30. Fooling SHAP with Stealthily Biased Sampling**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15419v1)

**Authors**: Gabriel Laberge, Ulrich Aïvodji, Satoshi Hara

**Abstracts**: SHAP explanations aim at identifying which features contribute the most to the difference in model prediction at a specific input versus a background distribution. Recent studies have shown that they can be manipulated by malicious adversaries to produce arbitrary desired explanations. However, existing attacks focus solely on altering the black-box model itself. In this paper, we propose a complementary family of attacks that leave the model intact and manipulate SHAP explanations using stealthily biased sampling of the data points used to approximate expectations w.r.t the background distribution. In the context of fairness audit, we show that our attack can reduce the importance of a sensitive feature when explaining the difference in outcomes between groups, while remaining undetected. These results highlight the manipulability of SHAP explanations and encourage auditors to treat post-hoc explanations with skepticism.



## **31. Searching for the Essence of Adversarial Perturbations**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15357v1)

**Authors**: Dennis Y. Menn, Hung-yi Lee

**Abstracts**: Neural networks have achieved the state-of-the-art performance on various machine learning fields, yet the incorporation of malicious perturbations with input data (adversarial example) is able to fool neural networks' predictions. This would lead to potential risks in real-world applications, for example, auto piloting and facial recognition. However, the reason for the existence of adversarial examples remains controversial. Here we demonstrate that adversarial perturbations contain human-recognizable information, which is the key conspirator responsible for a neural network's erroneous prediction. This concept of human-recognizable information allows us to explain key features related to adversarial perturbations, which include the existence of adversarial examples, the transferability among different neural networks, and the increased neural network interpretability for adversarial training. Two unique properties in adversarial perturbations that fool neural networks are uncovered: masking and generation. A special class, the complementary class, is identified when neural networks classify input images. The human-recognizable information contained in adversarial perturbations allows researchers to gain insight on the working principles of neural networks and may lead to develop techniques that detect/defense adversarial attacks.



## **32. GAN-based Medical Image Small Region Forgery Detection via a Two-Stage Cascade Framework**

eess.IV

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15170v1)

**Authors**: Jianyi Zhang, Xuanxi Huang, Yaqi Liu, Yuyang Han, Zixiao Xiang

**Abstracts**: Using generative adversarial network (GAN)\cite{RN90} for data enhancement of medical images is significantly helpful for many computer-aided diagnosis (CAD) tasks. A new attack called CT-GAN has emerged. It can inject or remove lung cancer lesions to CT scans. Because the tampering region may even account for less than 1\% of the original image, even state-of-the-art methods are challenging to detect the traces of such tampering.   This paper proposes a cascade framework to detect GAN-based medical image small region forgery like CT-GAN. In the local detection stage, we train the detector network with small sub-images so that interference information in authentic regions will not affect the detector. We use depthwise separable convolution and residual to prevent the detector from over-fitting and enhance the ability to find forged regions through the attention mechanism. The detection results of all sub-images in the same image will be combined into a heatmap. In the global classification stage, using gray level co-occurrence matrix (GLCM) can better extract features of the heatmap. Because the shape and size of the tampered area are uncertain, we train PCA and SVM methods for classification. Our method can classify whether a CT image has been tampered and locate the tampered position. Sufficient experiments show that our method can achieve excellent performance.



## **33. Why Adversarial Training of ReLU Networks Is Difficult?**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15130v1)

**Authors**: Xu Cheng, Hao Zhang, Yue Xin, Wen Shen, Jie Ren, Quanshi Zhang

**Abstracts**: This paper mathematically derives an analytic solution of the adversarial perturbation on a ReLU network, and theoretically explains the difficulty of adversarial training. Specifically, we formulate the dynamics of the adversarial perturbation generated by the multi-step attack, which shows that the adversarial perturbation tends to strengthen eigenvectors corresponding to a few top-ranked eigenvalues of the Hessian matrix of the loss w.r.t. the input. We also prove that adversarial training tends to strengthen the influence of unconfident input samples with large gradient norms in an exponential manner. Besides, we find that adversarial training strengthens the influence of the Hessian matrix of the loss w.r.t. network parameters, which makes the adversarial training more likely to oscillate along directions of a few samples, and boosts the difficulty of adversarial training. Crucially, our proofs provide a unified explanation for previous findings in understanding adversarial training.



## **34. Domain Constraints in Feature Space: Strengthening Robustness of Android Malware Detection against Realizable Adversarial Examples**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15128v1)

**Authors**: Hamid Bostani, Zhuoran Liu, Zhengyu Zhao, Veelasha Moonsamy

**Abstracts**: Strengthening the robustness of machine learning-based malware detectors against realistic evasion attacks remains one of the major obstacles for Android malware detection. To that end, existing work has focused on interpreting domain constraints of Android malware in the problem space, where problem-space realizable adversarial examples are generated. In this paper, we provide another promising way to achieve the same goal but based on interpreting the domain constraints in the feature space, where feature-space realizable adversarial examples are generated. Specifically, we present a novel approach to extracting feature-space domain constraints by learning meaningful feature dependencies from data, and applying them based on a novel robust feature space. Experimental results successfully demonstrate the effectiveness of our novel robust feature space in providing adversarial robustness for DREBIN, a state-of-the-art Android malware detector. For example, it can decrease the evasion rate of a realistic gradient-based attack by $96.4\%$ in a limited-knowledge (transfer) setting and by $13.8\%$ in a more challenging, perfect-knowledge setting. In addition, we show that directly using our learned domain constraints in the adversarial retraining framework leads to about $84\%$ improvement in a limited-knowledge setting, with up to $377\times$ faster implementation than using problem-space adversarial examples.



## **35. Guided Diffusion Model for Adversarial Purification**

cs.CV

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14969v1)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.



## **36. CalFAT: Calibrated Federated Adversarial Training with Label Skewness**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14926v1)

**Authors**: Chen Chen, Yuchen Liu, Xingjun Ma, Lingjuan Lyu

**Abstracts**: Recent studies have shown that, like traditional machine learning, federated learning (FL) is also vulnerable to adversarial attacks. To improve the adversarial robustness of FL, few federated adversarial training (FAT) methods have been proposed to apply adversarial training locally before global aggregation. Although these methods demonstrate promising results on independent identically distributed (IID) data, they suffer from training instability issues on non-IID data with label skewness, resulting in much degraded natural accuracy. This tends to hinder the application of FAT in real-world applications where the label distribution across the clients is often skewed. In this paper, we study the problem of FAT under label skewness, and firstly reveal one root cause of the training instability and natural accuracy degradation issues: skewed labels lead to non-identical class probabilities and heterogeneous local models. We then propose a Calibrated FAT (CalFAT) approach to tackle the instability issue by calibrating the logits adaptively to balance the classes. We show both theoretically and empirically that the optimization of CalFAT leads to homogeneous local models across the clients and much improved convergence rate and final performance.



## **37. CausalAdv: Adversarial Robustness through the Lens of Causality**

cs.LG

ICLR2022, 20 pages, 3 figures

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2106.06196v2)

**Authors**: Yonggang Zhang, Mingming Gong, Tongliang Liu, Gang Niu, Xinmei Tian, Bo Han, Bernhard Schölkopf, Kun Zhang

**Abstracts**: The adversarial vulnerability of deep neural networks has attracted significant attention in machine learning. As causal reasoning has an instinct for modelling distribution change, it is essential to incorporate causality into analyzing this specific type of distribution change induced by adversarial attacks. However, causal formulations of the intuition of adversarial attacks and the development of robust DNNs are still lacking in the literature. To bridge this gap, we construct a causal graph to model the generation process of adversarial examples and define the adversarial distribution to formalize the intuition of adversarial attacks. From the causal perspective, we study the distinction between the natural and adversarial distribution and conclude that the origin of adversarial vulnerability is the focus of models on spurious correlations. Inspired by the causal understanding, we propose the Causal inspired Adversarial distribution alignment method, CausalAdv, to eliminate the difference between natural and adversarial distributions by considering spurious correlations. Extensive experiments demonstrate the efficacy of the proposed method. Our work is the first attempt towards using causality to understand and mitigate the adversarial vulnerability.



## **38. Exposing Fine-grained Adversarial Vulnerability of Face Anti-spoofing Models**

cs.CV

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14851v1)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Bo Peng, Jing Dong

**Abstracts**: Adversarial attacks seriously threaten the high accuracy of face anti-spoofing models. Little adversarial noise can perturb their classification of live and spoofing. The existing adversarial attacks fail to figure out which part of the target face anti-spoofing model is vulnerable, making adversarial analysis tricky. So we propose fine-grained attacks for exposing adversarial vulnerability of face anti-spoofing models. Firstly, we propose Semantic Feature Augmentation (SFA) module, which makes adversarial noise semantic-aware to live and spoofing features. SFA considers the contrastive classes of data and texture bias of models in the context of face anti-spoofing, increasing the attack success rate by nearly 40% on average. Secondly, we generate fine-grained adversarial examples based on SFA and the multitask network with auxiliary information. We evaluate three annotations (facial attributes, spoofing types and illumination) and two geometric maps (depth and reflection), on four backbone networks (VGG, Resnet, Densenet and Swin Transformer). We find that facial attributes annotation and state-of-art networks fail to guarantee that models are robust to adversarial attacks. Such adversarial attacks can be generalized to more auxiliary information and backbone networks, to help our community handle the trade-off between accuracy and adversarial robustness.



## **39. Efficient Reward Poisoning Attacks on Online Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14842v1)

**Authors**: Yinglun Xu, Qi Zeng, Gagandeep Singh

**Abstracts**: We study data poisoning attacks on online deep reinforcement learning (DRL) where the attacker is oblivious to the learning algorithm used by the agent and does not necessarily have full knowledge of the environment. We demonstrate the intrinsic vulnerability of state-of-the-art DRL algorithms by designing a general reward poisoning framework called adversarial MDP attacks. We instantiate our framework to construct several new attacks which only corrupt the rewards for a small fraction of the total training timesteps and make the agent learn a low-performing policy. Our key insight is that the state-of-the-art DRL algorithms strategically explore the environment to find a high-performing policy. Our attacks leverage this insight to construct a corrupted environment for misleading the agent towards learning low-performing policies with a limited attack budget. We provide a theoretical analysis of the efficiency of our attack and perform an extensive evaluation. Our results show that our attacks efficiently poison agents learning with a variety of state-of-the-art DRL algorithms, such as DQN, PPO, SAC, etc. under several popular classical control and MuJoCo environments.



## **40. Mixture GAN For Modulation Classification Resiliency Against Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.15743v1)

**Authors**: Eyad Shtaiwi, Ahmed El Ouadrhiri, Majid Moradikia, Salma Sultana, Ahmed Abdelhadi, Zhu Han

**Abstracts**: Automatic modulation classification (AMC) using the Deep Neural Network (DNN) approach outperforms the traditional classification techniques, even in the presence of challenging wireless channel environments. However, the adversarial attacks cause the loss of accuracy for the DNN-based AMC by injecting a well-designed perturbation to the wireless channels. In this paper, we propose a novel generative adversarial network (GAN)-based countermeasure approach to safeguard the DNN-based AMC systems against adversarial attack examples. GAN-based aims to eliminate the adversarial attack examples before feeding to the DNN-based classifier. Specifically, we have shown the resiliency of our proposed defense GAN against the Fast-Gradient Sign method (FGSM) algorithm as one of the most potent kinds of attack algorithms to craft the perturbed signals. The existing defense-GAN has been designed for image classification and does not work in our case where the above-mentioned communication system is considered. Thus, our proposed countermeasure approach deploys GANs with a mixture of generators to overcome the mode collapsing problem in a typical GAN facing radio signal classification problem. Simulation results show the effectiveness of our proposed defense GAN so that it could enhance the accuracy of the DNN-based AMC under adversarial attacks to 81%, approximately.



## **41. Unfooling Perturbation-Based Post Hoc Explainers**

cs.AI

10 pages (not including references and supplemental)

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14772v1)

**Authors**: Zachariah Carmichael, Walter J Scheirer

**Abstracts**: Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP.



## **42. On the Robustness of Safe Reinforcement Learning under Observational Perturbations**

cs.LG

27 pages, 3 figures, 3 tables

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14691v1)

**Authors**: Zuxin Liu, Zijian Guo, Zhepeng Cen, Huan Zhang, Jie Tan, Bo Li, Ding Zhao

**Abstracts**: Safe reinforcement learning (RL) trains a policy to maximize the task reward while satisfying safety constraints. While prior works focus on the performance optimality, we find that the optimal solutions of many safe RL problems are not robust and safe against carefully designed observational perturbations. We formally analyze the unique properties of designing effective state adversarial attackers in the safe RL setting. We show that baseline adversarial attack techniques for standard RL tasks are not always effective for safe RL and proposed two new approaches - one maximizes the cost and the other maximizes the reward. One interesting and counter-intuitive finding is that the maximum reward attack is strong, as it can both induce unsafe behaviors and make the attack stealthy by maintaining the reward. We further propose a more effective adversarial training framework for safe RL and evaluate it via comprehensive experiments. This work sheds light on the inherited connection between observational robustness and safety in RL and provides a pioneer work for future safe RL studies.



## **43. Superclass Adversarial Attack**

cs.CV

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14629v1)

**Authors**: Soichiro Kumano, Hiroshi Kera, Toshihiko Yamasaki

**Abstracts**: Adversarial attacks have only focused on changing the predictions of the classifier, but their danger greatly depends on how the class is mistaken. For example, when an automatic driving system mistakes a Persian cat for a Siamese cat, it is hardly a problem. However, if it mistakes a cat for a 120km/h minimum speed sign, serious problems can arise. As a stepping stone to more threatening adversarial attacks, we consider the superclass adversarial attack, which causes misclassification of not only fine classes, but also superclasses. We conducted the first comprehensive analysis of superclass adversarial attacks (an existing and 19 new methods) in terms of accuracy, speed, and stability, and identified several strategies to achieve better performance. Although this study is aimed at superclass misclassification, the findings can be applied to other problem settings involving multiple classes, such as top-k and multi-label classification attacks.



## **44. Graph Structure Based Data Augmentation Method**

cs.LG

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14619v1)

**Authors**: Kyung Geun Kim, Byeong Tak Lee

**Abstracts**: In this paper, we propose a novel graph-based data augmentation method that can generally be applied to medical waveform data with graph structures. In the process of recording medical waveform data, such as electrocardiogram (ECG) or electroencephalogram (EEG), angular perturbations between the measurement leads exist due to discrepancies in lead positions. The data samples with large angular perturbations often cause inaccuracy in algorithmic prediction tasks. We design a graph-based data augmentation technique that exploits the inherent graph structures within the medical waveform data to improve both performance and robustness. In addition, we show that the performance gain from graph augmentation results from robustness by testing against adversarial attacks. Since the bases of performance gain are orthogonal, the graph augmentation can be used in conjunction with existing data augmentation techniques to further improve the final performance. We believe that our graph augmentation method opens up new possibilities to explore in data augmentation.



## **45. Problem-Space Evasion Attacks in the Android OS: a Survey**

cs.CR

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14576v1)

**Authors**: Harel Berger, Dr. Chen Hajaj, Dr. Amit Dvir

**Abstracts**: Android is the most popular OS worldwide. Therefore, it is a target for various kinds of malware. As a countermeasure, the security community works day and night to develop appropriate Android malware detection systems, with ML-based or DL-based systems considered as some of the most common types. Against these detection systems, intelligent adversaries develop a wide set of evasion attacks, in which an attacker slightly modifies a malware sample to evade its target detection system. In this survey, we address problem-space evasion attacks in the Android OS, where attackers manipulate actual APKs, rather than their extracted feature vector. We aim to explore this kind of attacks, frequently overlooked by the research community due to a lack of knowledge of the Android domain, or due to focusing on general mathematical evasion attacks - i.e., feature-space evasion attacks. We discuss the different aspects of problem-space evasion attacks, using a new taxonomy, which focuses on key ingredients of each problem-space attack, such as the attacker model, the attacker's mode of operation, and the functional assessment of post-attack applications.



## **46. BadDet: Backdoor Attacks on Object Detection**

cs.CV

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2205.14497v1)

**Authors**: Shih-Han Chan, Yinpeng Dong, Jun Zhu, Xiaolu Zhang, Jun Zhou

**Abstracts**: Deep learning models have been deployed in numerous real-world applications such as autonomous driving and surveillance. However, these models are vulnerable in adversarial environments. Backdoor attack is emerging as a severe security threat which injects a backdoor trigger into a small portion of training data such that the trained model behaves normally on benign inputs but gives incorrect predictions when the specific trigger appears. While most research in backdoor attacks focuses on image classification, backdoor attacks on object detection have not been explored but are of equal importance. Object detection has been adopted as an important module in various security-sensitive applications such as autonomous driving. Therefore, backdoor attacks on object detection could pose severe threats to human lives and properties. We propose four kinds of backdoor attacks for object detection task: 1) Object Generation Attack: a trigger can falsely generate an object of the target class; 2) Regional Misclassification Attack: a trigger can change the prediction of a surrounding object to the target class; 3) Global Misclassification Attack: a single trigger can change the predictions of all objects in an image to the target class; and 4) Object Disappearance Attack: a trigger can make the detector fail to detect the object of the target class. We develop appropriate metrics to evaluate the four backdoor attacks on object detection. We perform experiments using two typical object detection models -- Faster-RCNN and YOLOv3 on different datasets. More crucially, we demonstrate that even fine-tuning on another benign dataset cannot remove the backdoor hidden in the object detection model. To defend against these backdoor attacks, we propose Detector Cleanse, an entropy-based run-time detection framework to identify poisoned testing samples for any deployed object detector.



## **47. Policy Smoothing for Provably Robust Reinforcement Learning**

cs.LG

Published as a conference paper at ICLR 2022

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2106.11420v3)

**Authors**: Aounon Kumar, Alexander Levine, Soheil Feizi

**Abstracts**: The study of provable adversarial robustness for deep neural networks (DNNs) has mainly focused on static supervised learning tasks such as image classification. However, DNNs have been used extensively in real-world adaptive tasks such as reinforcement learning (RL), making such systems vulnerable to adversarial attacks as well. Prior works in provable robustness in RL seek to certify the behaviour of the victim policy at every time-step against a non-adaptive adversary using methods developed for the static setting. But in the real world, an RL adversary can infer the defense strategy used by the victim agent by observing the states, actions, etc., from previous time-steps and adapt itself to produce stronger attacks in future steps. We present an efficient procedure, designed specifically to defend against an adaptive RL adversary, that can directly certify the total reward without requiring the policy to be robust at each time-step. Our main theoretical contribution is to prove an adaptive version of the Neyman-Pearson Lemma -- a key lemma for smoothing-based certificates -- where the adversarial perturbation at a particular time can be a stochastic function of current and previous observations and states as well as previous actions. Building on this result, we propose policy smoothing where the agent adds a Gaussian noise to its observation at each time-step before passing it through the policy function. Our robustness certificates guarantee that the final total reward obtained by policy smoothing remains above a certain threshold, even though the actions at intermediate time-steps may change under the attack. Our experiments on various environments like Cartpole, Pong, Freeway and Mountain Car show that our method can yield meaningful robustness guarantees in practice.



## **48. Certifying Model Accuracy under Distribution Shifts**

cs.LG

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2201.12440v2)

**Authors**: Aounon Kumar, Alexander Levine, Tom Goldstein, Soheil Feizi

**Abstracts**: Certified robustness in machine learning has primarily focused on adversarial perturbations of the input with a fixed attack budget for each point in the data distribution. In this work, we present provable robustness guarantees on the accuracy of a model under bounded Wasserstein shifts of the data distribution. We show that a simple procedure that randomizes the input of the model within a transformation space is provably robust to distributional shifts under the transformation. Our framework allows the datum-specific perturbation size to vary across different points in the input distribution and is general enough to include fixed-sized perturbations as well. Our certificates produce guaranteed lower bounds on the performance of the model for any (natural or adversarial) shift of the input distribution within a Wasserstein ball around the original distribution. We apply our technique to: (i) certify robustness against natural (non-adversarial) transformations of images such as color shifts, hue shifts and changes in brightness and saturation, (ii) certify robustness against adversarial shifts of the input distribution, and (iii) show provable lower bounds (hardness results) on the performance of models trained on so-called "unlearnable" datasets that have been poisoned to interfere with model training.



## **49. SHORTSTACK: Distributed, Fault-tolerant, Oblivious Data Access**

cs.CR

Full version of USENIX OSDI'22 paper

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2205.14281v1)

**Authors**: Midhul Vuppalapati, Kushal Babel, Anurag Khandelwal, Rachit Agarwal

**Abstracts**: Many applications that benefit from data offload to cloud services operate on private data. A now-long line of work has shown that, even when data is offloaded in an encrypted form, an adversary can learn sensitive information by analyzing data access patterns. Existing techniques for oblivious data access--that protect against access pattern attacks--require a centralized, stateful and trusted, proxy to orchestrate data accesses from applications to cloud services. We show that, in failure-prone deployments, such a centralized and stateful proxy results in violation of oblivious data access security guarantees and/or system unavailability. Thus, we initiate the study of distributed, fault-tolerant, oblivious data access.   We present SHORTSTACK, a distributed proxy architecture for oblivious data access in failure-prone deployments. SHORTSTACK achieves the classical obliviousness guarantee--access patterns observed by the adversary being independent of the input--even under a powerful passive persistent adversary that can force failure of arbitrary (bounded-sized) subset of proxy servers at arbitrary times. We also introduce a security model that enables studying oblivious data access with distributed, failure-prone, servers. We provide a formal proof that SHORTSTACK enables oblivious data access under this model, and show empirically that SHORTSTACK performance scales near-linearly with number of distributed proxy servers.



## **50. Semi-supervised Semantics-guided Adversarial Training for Trajectory Prediction**

cs.LG

11 pages, adversarial training for trajectory prediction

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.14230v1)

**Authors**: Ruochen Jiao, Xiangguo Liu, Takami Sato, Qi Alfred Chen, Qi Zhu

**Abstracts**: Predicting the trajectories of surrounding objects is a critical task in self-driving and many other autonomous systems. Recent works demonstrate that adversarial attacks on trajectory prediction, where small crafted perturbations are introduced to history trajectories, may significantly mislead the prediction of future trajectories and ultimately induce unsafe planning. However, few works have addressed enhancing the robustness of this important safety-critical task. In this paper, we present the first adversarial training method for trajectory prediction. Compared with typical adversarial training on image tasks, our work is challenged by more random inputs with rich context, and a lack of class labels. To address these challenges, we propose a method based on a semi-supervised adversarial autoencoder that models disentangled semantic features with domain knowledge and provides additional latent labels for the adversarial training. Extensive experiments with different types of attacks demonstrate that our semi-supervised semantics-guided adversarial training method can effectively mitigate the impact of adversarial attacks and generally improve the system's adversarial robustness to a variety of attacks, including unseen ones. We believe that such semantics-guided architecture and advancement in robust generalization is an important step for developing robust prediction models and enabling safe decision making.



