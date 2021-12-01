# Latest Adversarial Attack Papers
**update at 2021-12-01 23:56:48**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Trustworthy Medical Segmentation with Uncertainty Estimation**

eess.IV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.05978v2)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Ghulam Rasool, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.



## **2. Defending Against Adversarial Denial-of-Service Data Poisoning Attacks**

cs.CR

Published at ACSAC DYNAMICS 2020

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2104.06744v3)

**Authors**: Nicolas M. Müller, Simon Roschmann, Konstantin Böttinger

**Abstracts**: Data poisoning is one of the most relevant security threats against machine learning and data-driven technologies. Since many applications rely on untrusted training data, an attacker can easily craft malicious samples and inject them into the training dataset to degrade the performance of machine learning models. As recent work has shown, such Denial-of-Service (DoS) data poisoning attacks are highly effective. To mitigate this threat, we propose a new approach of detecting DoS poisoned instances. In comparison to related work, we deviate from clustering and anomaly detection based approaches, which often suffer from the curse of dimensionality and arbitrary anomaly threshold selection. Rather, our defence is based on extracting information from the training data in such a generalized manner that we can identify poisoned samples based on the information present in the unpoisoned portion of the data. We evaluate our defence against two DoS poisoning attacks and seven datasets, and find that it reliably identifies poisoned instances. In comparison to related work, our defence improves false positive / false negative rates by at least 50%, often more.



## **3. FROB: Few-shot ROBust Model for Classification and Out-of-Distribution Detection**

cs.LG

Paper, 22 pages, Figures, Tables

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15487v1)

**Authors**: Nikolaos Dionelis

**Abstracts**: Nowadays, classification and Out-of-Distribution (OoD) detection in the few-shot setting remain challenging aims due to rarity and the limited samples in the few-shot setting, and because of adversarial attacks. Accomplishing these aims is important for critical systems in safety, security, and defence. In parallel, OoD detection is challenging since deep neural network classifiers set high confidence to OoD samples away from the training data. To address such limitations, we propose the Few-shot ROBust (FROB) model for classification and few-shot OoD detection. We devise FROB for improved robustness and reliable confidence prediction for few-shot OoD detection. We generate the support boundary of the normal class distribution and combine it with few-shot Outlier Exposure (OE). We propose a self-supervised learning few-shot confidence boundary methodology based on generative and discriminative models. The contribution of FROB is the combination of the generated boundary in a self-supervised learning manner and the imposition of low confidence at this learned boundary. FROB implicitly generates strong adversarial samples on the boundary and forces samples from OoD, including our boundary, to be less confident by the classifier. FROB achieves generalization to unseen OoD with applicability to unknown, in the wild, test sets that do not correlate to the training datasets. To improve robustness, FROB redesigns OE to work even for zero-shots. By including our boundary, FROB reduces the threshold linked to the model's few-shot robustness; it maintains the OoD performance approximately independent of the number of few-shots. The few-shot robustness analysis evaluation of FROB on different sets and on One-Class Classification (OCC) data shows that FROB achieves competitive performance and outperforms benchmarks in terms of robustness to the outlier few-shot sample population and variability.



## **4. A Face Recognition System's Worst Morph Nightmare, Theoretically**

cs.CV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15416v1)

**Authors**: Una M. Kelly, Raymond Veldhuis, Luuk Spreeuwers

**Abstracts**: It has been shown that Face Recognition Systems (FRSs) are vulnerable to morphing attacks, but most research focusses on landmark-based morphs. A second method for generating morphs uses Generative Adversarial Networks, which results in convincingly real facial images that can be almost as challenging for FRSs as landmark-based attacks. We propose a method to create a third, different type of morph, that has the advantage of being easier to train. We introduce the theoretical concept of \textit{worst-case morphs}, which are those morphs that are most challenging for a fixed FRS. For a set of images and corresponding embeddings in an FRS's latent space, we generate images that approximate these worst-case morphs using a mapping from embedding space back to image space. While the resulting images are not yet as challenging as other morphs, they can provide valuable information in future research on Morphing Attack Detection (MAD) methods and on weaknesses of FRSs. Methods for MAD need to be validated on more varied morph databases. Our proposed method contributes to achieving such variation.



## **5. Medical Aegis: Robust adversarial protectors for medical images**

cs.CV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.10969v2)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.



## **6. COREATTACK: Breaking Up the Core Structure of Graphs**

cs.SI

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15276v1)

**Authors**: Bo Zhou, Yuqian Lv, Jinhuan Wang, Jian Zhang, Qi Xuan

**Abstracts**: The concept of k-core in complex networks plays a key role in many applications, e.g., understanding the global structure, or identifying central/critical nodes, of a network. A malicious attacker with jamming ability can exploit the vulnerability of the k-core structure to attack the network and invalidate the network analysis methods, e.g., reducing the k-shell values of nodes can deceive graph algorithms, leading to the wrong decisions. In this paper, we investigate the robustness of the k-core structure under adversarial attacks by deleting edges, for the first time. Firstly, we give the general definition of targeted k-core attack, map it to the set cover problem which is NP-hard, and further introduce a series of evaluation metrics to measure the performance of attack methods. Then, we propose $Q$ index theoretically as the probability that the terminal node of an edge does not belong to the innermost core, which is further used to guide the design of our heuristic attack methods, namely COREATTACK and GreedyCOREATTACK. The experiments on a variety of real-world networks demonstrate that our methods behave much better than a series of baselines, in terms of much smaller Edge Change Rate (ECR) and False Attack Rate (FAR), achieving state-of-the-art attack performance. More impressively, for certain real-world networks, only deleting one edge from the k-core may lead to the collapse of the innermost core, even if this core contains dozens of nodes. Such a phenomenon indicates that the k-core structure could be extremely vulnerable under adversarial attacks, and its robustness thus should be carefully addressed to ensure the security of many graph algorithms.



## **7. Black-box Adversarial Attacks on Commercial Speech Platforms with Minimal Information**

cs.CR

A version of this paper appears in the proceedings of the 28th ACM  Conference on Computer and Communications Security (CCS 2021). The notes in  Tables 1 and 4 have been updated

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2110.09714v2)

**Authors**: Baolin Zheng, Peipei Jiang, Qian Wang, Qi Li, Chao Shen, Cong Wang, Yunjie Ge, Qingyang Teng, Shenyi Zhang

**Abstracts**: Adversarial attacks against commercial black-box speech platforms, including cloud speech APIs and voice control devices, have received little attention until recent years. The current "black-box" attacks all heavily rely on the knowledge of prediction/confidence scores to craft effective adversarial examples, which can be intuitively defended by service providers without returning these messages. In this paper, we propose two novel adversarial attacks in more practical and rigorous scenarios. For commercial cloud speech APIs, we propose Occam, a decision-only black-box adversarial attack, where only final decisions are available to the adversary. In Occam, we formulate the decision-only AE generation as a discontinuous large-scale global optimization problem, and solve it by adaptively decomposing this complicated problem into a set of sub-problems and cooperatively optimizing each one. Our Occam is a one-size-fits-all approach, which achieves 100% success rates of attacks with an average SNR of 14.23dB, on a wide range of popular speech and speaker recognition APIs, including Google, Alibaba, Microsoft, Tencent, iFlytek, and Jingdong, outperforming the state-of-the-art black-box attacks. For commercial voice control devices, we propose NI-Occam, the first non-interactive physical adversarial attack, where the adversary does not need to query the oracle and has no access to its internal information and training data. We combine adversarial attacks with model inversion attacks, and thus generate the physically-effective audio AEs with high transferability without any interaction with target devices. Our experimental results show that NI-Occam can successfully fool Apple Siri, Microsoft Cortana, Google Assistant, iFlytek and Amazon Echo with an average SRoA of 52% and SNR of 9.65dB, shedding light on non-interactive physical attacks against voice control devices.



## **8. Mitigating Adversarial Attacks by Distributing Different Copies to Different Users**

cs.CR

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15160v1)

**Authors**: Jiyi Zhang, Wesley Joon-Wie Tann, Ee-Chien Chang

**Abstracts**: Machine learning models are vulnerable to adversarial attacks. In this paper, we consider the scenario where a model is to be distributed to multiple users, among which a malicious user attempts to attack another user. The malicious user probes its copy of the model to search for adversarial samples and then presents the found samples to the victim's model in order to replicate the attack. We point out that by distributing different copies of the model to different users, we can mitigate the attack such that adversarial samples found on one copy would not work on another copy. We first observed that training a model with different randomness indeed mitigates such replication to certain degree. However, there is no guarantee and retraining is computationally expensive. Next, we propose a flexible parameter rewriting method that directly modifies the model's parameters. This method does not require additional training and is able to induce different sets of adversarial samples in different copies in a more controllable manner. Experimentation studies show that our approach can significantly mitigate the attacks while retaining high classification accuracy. From this study, we believe that there are many further directions worth exploring.



## **9. Adversarial Robustness of Deep Code Comment Generation**

cs.SE

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2108.00213v3)

**Authors**: Yu Zhou, Xiaoqing Zhang, Juanjuan Shen, Tingting Han, Taolue Chen, Harald Gall

**Abstracts**: Deep neural networks (DNNs) have shown remarkable performance in a variety of domains such as computer vision, speech recognition, or natural language processing. Recently they also have been applied to various software engineering tasks, typically involving processing source code. DNNs are well-known to be vulnerable to adversarial examples, i.e., fabricated inputs that could lead to various misbehaviors of the DNN model while being perceived as benign by humans. In this paper, we focus on the code comment generation task in software engineering and study the robustness issue of the DNNs when they are applied to this task. We propose ACCENT, an identifier substitution approach to craft adversarial code snippets, which are syntactically correct and semantically close to the original code snippet, but may mislead the DNNs to produce completely irrelevant code comments. In order to improve the robustness, ACCENT also incorporates a novel training method, which can be applied to existing code comment generation models. We conduct comprehensive experiments to evaluate our approach by attacking the mainstream encoder-decoder architectures on two large-scale publicly available datasets. The results show that ACCENT efficiently produces stable attacks with functionality-preserving adversarial examples, and the generated examples have better transferability compared with baselines. We also confirm, via experiments, the effectiveness in improving model robustness with our training method.



## **10. Living-Off-The-Land Command Detection Using Active Learning**

cs.CR

14 pages, published in RAID 2021

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15039v1)

**Authors**: Talha Ongun, Jack W. Stokes, Jonathan Bar Or, Ke Tian, Farid Tajaddodianfar, Joshua Neil, Christian Seifert, Alina Oprea, John C. Platt

**Abstracts**: In recent years, enterprises have been targeted by advanced adversaries who leverage creative ways to infiltrate their systems and move laterally to gain access to critical data. One increasingly common evasive method is to hide the malicious activity behind a benign program by using tools that are already installed on user computers. These programs are usually part of the operating system distribution or another user-installed binary, therefore this type of attack is called "Living-Off-The-Land". Detecting these attacks is challenging, as adversaries may not create malicious files on the victim computers and anti-virus scans fail to detect them. We propose the design of an Active Learning framework called LOLAL for detecting Living-Off-the-Land attacks that iteratively selects a set of uncertain and anomalous samples for labeling by a human analyst. LOLAL is specifically designed to work well when a limited number of labeled samples are available for training machine learning models to detect attacks. We investigate methods to represent command-line text using word-embedding techniques, and design ensemble boosting classifiers to distinguish malicious and benign samples based on the embedding representation. We leverage a large, anonymized dataset collected by an endpoint security product and demonstrate that our ensemble classifiers achieve an average F1 score of 0.96 at classifying different attack classes. We show that our active learning method consistently improves the classifier performance, as more training data is labeled, and converges in less than 30 iterations when starting with a small number of labeled instances.



## **11. Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training**

cs.LG

Accepted to ICML 2021 Adversarial Machine Learning Workshop. Under  review

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2012.12368v4)

**Authors**: Theodoros Tsiligkaridis, Jay Roberts

**Abstracts**: Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense against such attacks. Due to the high computation time for generating strong adversarial examples for AT, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training. Although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical framework for adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the $\ell_2$ distortion of $\ell_\infty$ FW attacks. We analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that $\ell_\infty$ attacks against robust models achieve near maximal $\ell_2$ distortion, while standard networks have lower distortion. Furthermore, it is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps to increase efficiency without compromising robustness. FW-AT-Adapt provides training times on par with single-step fast AT methods and improves closing the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.



## **12. MedRDF: A Robust and Retrain-Less Diagnostic Framework for Medical Pretrained Models Against Adversarial Attack**

cs.CV

TMI under review

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2111.14564v1)

**Authors**: Mengting Xu, Tao Zhang, Daoqiang Zhang

**Abstracts**: Deep neural networks are discovered to be non-robust when attacked by imperceptible adversarial examples, which is dangerous for it applied into medical diagnostic system that requires high reliability. However, the defense methods that have good effect in natural images may not be suitable for medical diagnostic tasks. The preprocessing methods (e.g., random resizing, compression) may lead to the loss of the small lesions feature in the medical image. Retraining the network on the augmented data set is also not practical for medical models that have already been deployed online. Accordingly, it is necessary to design an easy-to-deploy and effective defense framework for medical diagnostic tasks. In this paper, we propose a Robust and Retrain-Less Diagnostic Framework for Medical pretrained models against adversarial attack (i.e., MedRDF). It acts on the inference time of the pertained medical model. Specifically, for each test image, MedRDF firstly creates a large number of noisy copies of it, and obtains the output labels of these copies from the pretrained medical diagnostic model. Then, based on the labels of these copies, MedRDF outputs the final robust diagnostic result by majority voting. In addition to the diagnostic result, MedRDF produces the Robust Metric (RM) as the confidence of the result. Therefore, it is convenient and reliable to utilize MedRDF to convert pre-trained non-robust diagnostic models into robust ones. The experimental results on COVID-19 and DermaMNIST datasets verify the effectiveness of our MedRDF in improving the robustness of medical diagnostic models.



## **13. Reliably fast adversarial training via latent adversarial perturbation**

cs.LG

ICCV 2021 (Oral)

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2104.01575v2)

**Authors**: Geon Yeong Park, Sang Wan Lee

**Abstracts**: While multi-step adversarial training is widely popular as an effective defense method against strong adversarial attacks, its computational cost is notoriously expensive, compared to standard training. Several single-step adversarial training methods have been proposed to mitigate the above-mentioned overhead cost; however, their performance is not sufficiently reliable depending on the optimization setting. To overcome such limitations, we deviate from the existing input-space-based adversarial training regime and propose a single-step latent adversarial training method (SLAT), which leverages the gradients of latent representation as the latent adversarial perturbation. We demonstrate that the L1 norm of feature gradients is implicitly regularized through the adopted latent perturbation, thereby recovering local linearity and ensuring reliable performance, compared to the existing single-step adversarial training methods. Because latent perturbation is based on the gradients of the latent representations which can be obtained for free in the process of input gradients computation, the proposed method costs roughly the same time as the fast gradient sign method. Experiment results demonstrate that the proposed method, despite its structural simplicity, outperforms state-of-the-art accelerated adversarial training methods.



## **14. Adversarial Attacks in Cooperative AI**

cs.LG

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2111.14833v1)

**Authors**: Ted Fujimoto, Arthur Paul Pedersen

**Abstracts**: Single-agent reinforcement learning algorithms in a multi-agent environment are inadequate for fostering cooperation. If intelligent agents are to interact and work together to solve complex problems, methods that counter non-cooperative behavior are needed to facilitate the training of multiple agents. This is the goal of cooperative AI. Recent work in adversarial machine learning, however, shows that models (e.g., image classifiers) can be easily deceived into making incorrect decisions. In addition, some past research in cooperative AI has relied on new notions of representations, like public beliefs, to accelerate the learning of optimally cooperative behavior. Hence, cooperative AI might introduce new weaknesses not investigated in previous machine learning research. In this paper, our contributions include: (1) arguing that three algorithms inspired by human-like social intelligence introduce new vulnerabilities, unique to cooperative AI, that adversaries can exploit, and (2) an experiment showing that simple, adversarial perturbations on the agents' beliefs can negatively impact performance. This evidence points to the possibility that formal representations of social behavior are vulnerable to adversarial attacks.



## **15. Feature-Filter: Detecting Adversarial Examples through Filtering off Recessive Features**

cs.LG

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2107.09502v2)

**Authors**: Hui Liu, Bo Zhao, Minzhi Ji, Yuefeng Peng, Jiabao Guo, Peng Liu

**Abstracts**: Deep neural networks (DNNs) are under threat from adversarial example attacks. The adversary can easily change the outputs of DNNs by adding small well-designed perturbations to inputs. Adversarial example detection is a fundamental work for robust DNNs-based service. Adversarial examples show the difference between humans and DNNs in image recognition. From a human-centric perspective, image features could be divided into dominant features that are comprehensible to humans, and recessive features that are incomprehensible to humans, yet are exploited by DNNs. In this paper, we reveal that imperceptible adversarial examples are the product of recessive features misleading neural networks, and an adversarial attack is essentially a kind of method to enrich these recessive features in the image. The imperceptibility of the adversarial examples indicates that the perturbations enrich recessive features, yet hardly affect dominant features. Therefore, adversarial examples are sensitive to filtering off recessive features, while benign examples are immune to such operation. Inspired by this idea, we propose a label-only adversarial detection approach that is referred to as feature-filter. Feature-filter utilizes discrete cosine transform to approximately separate recessive features from dominant features, and gets a mutant image that is filtered off recessive features. By only comparing DNN's prediction labels on the input and its mutant, feature-filter can real-time detect imperceptible adversarial examples at high accuracy and few false positives.



## **16. GreedyFool: Multi-Factor Imperceptibility and Its Application to Designing a Black-box Adversarial Attack**

cs.LG

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2010.06855v4)

**Authors**: Hui Liu, Bo Zhao, Minzhi Ji, Peng Liu

**Abstracts**: Adversarial examples are well-designed input samples, in which perturbations are imperceptible to the human eyes, but easily mislead the output of deep neural networks (DNNs). Existing works synthesize adversarial examples by leveraging simple metrics to penalize perturbations, that lack sufficient consideration of the human visual system (HVS), which produces noticeable artifacts. To explore why the perturbations are visible, this paper summarizes four primary factors affecting the perceptibility of human eyes. Based on this investigation, we design a multi-factor metric MulFactorLoss for measuring the perceptual loss between benign examples and adversarial ones. In order to test the imperceptibility of the multi-factor metric, we propose a novel black-box adversarial attack that is referred to as GreedyFool. GreedyFool applies differential evolution to evaluate the effects of perturbed pixels on the confidence of a target DNN, and introduces greedy approximation to automatically generate adversarial perturbations. We conduct extensive experiments on the ImageNet and CIFRA-10 datasets and a comprehensive user study with 60 participants. The experimental results demonstrate that MulFactorLoss is a more imperceptible metric than the existing pixelwise metrics, and GreedyFool achieves a 100% success rate in a black-box manner.



## **17. Detecting Adversaries, yet Faltering to Noise? Leveraging Conditional Variational AutoEncoders for Adversary Detection in the Presence of Noisy Images**

cs.LG

**SubmitDate**: 2021-11-28    [paper-pdf](http://arxiv.org/pdf/2111.15518v1)

**Authors**: Dvij Kalaria, Aritra Hazra, Partha Pratim Chakrabarti

**Abstracts**: With the rapid advancement and increased use of deep learning models in image identification, security becomes a major concern to their deployment in safety-critical systems. Since the accuracy and robustness of deep learning models are primarily attributed from the purity of the training samples, therefore the deep learning architectures are often susceptible to adversarial attacks. Adversarial attacks are often obtained by making subtle perturbations to normal images, which are mostly imperceptible to humans, but can seriously confuse the state-of-the-art machine learning models. What is so special in the slightest intelligent perturbations or noise additions over normal images that it leads to catastrophic classifications by the deep neural networks? Using statistical hypothesis testing, we find that Conditional Variational AutoEncoders (CVAE) are surprisingly good at detecting imperceptible image perturbations. In this paper, we show how CVAEs can be effectively used to detect adversarial attacks on image classification networks. We demonstrate our results over MNIST, CIFAR-10 dataset and show how our method gives comparable performance to the state-of-the-art methods in detecting adversaries while not getting confused with noisy images, where most of the existing methods falter.



## **18. MALIGN: Adversarially Robust Malware Family Detection using Sequence Alignment**

cs.CR

**SubmitDate**: 2021-11-28    [paper-pdf](http://arxiv.org/pdf/2111.14185v1)

**Authors**: Shoumik Saha, Sadia Afroz, Atif Rahman

**Abstracts**: We propose MALIGN, a novel malware family detection approach inspired by genome sequence alignment. MALIGN encodes malware using four nucleotides and then uses genome sequence alignment approaches to create a signature of a malware family based on the code fragments conserved in the family making it robust to evasion by modification and addition of content. Moreover, unlike previous approaches based on sequence alignment, our method uses a multiple whole-genome alignment tool that protects against adversarial attacks such as code insertion, deletion or modification. Our approach outperforms state-of-the-art machine learning based malware detectors and demonstrates robustness against trivial adversarial attacks. MALIGN also helps identify the techniques malware authors use to evade detection.



## **19. Statically Detecting Adversarial Malware through Randomised Chaining**

cs.CR

**SubmitDate**: 2021-11-28    [paper-pdf](http://arxiv.org/pdf/2111.14037v1)

**Authors**: Matthew Crawford, Wei Wang, Ruoxi Sun, Minhui Xue

**Abstracts**: With the rapid growth of malware attacks, more antivirus developers consider deploying machine learning technologies into their productions. Researchers and developers published various machine learning-based detectors with high precision on malware detection in recent years. Although numerous machine learning-based malware detectors are available, they face various machine learning-targeted attacks, including evasion and adversarial attacks. This project explores how and why adversarial examples evade malware detectors, then proposes a randomised chaining method to defend against adversarial malware statically. This research is crucial for working towards combating the pertinent malware cybercrime.



## **20. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

cs.CV

11 pages, 8 figures, 10 tables

**SubmitDate**: 2021-11-27    [paper-pdf](http://arxiv.org/pdf/2005.09147v5)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Convolutional neural network (CNN) has surpassed traditional methods for medical image classification. However, CNN is vulnerable to adversarial attacks which may lead to disastrous consequences in medical applications. Although adversarial noises are usually generated by attack algorithms, white-noise-induced adversarial samples can exist, and therefore the threats are real. In this study, we propose a novel training method, named IMA, to improve the robust-ness of CNN against adversarial noises. During training, the IMA method increases the margins of training samples in the input space, i.e., moving CNN decision boundaries far away from the training samples to improve robustness. The IMA method is evaluated on publicly available datasets under strong 100-PGD white-box adversarial attacks, and the results show that the proposed method significantly improved CNN classification and segmentation accuracy on noisy data while keeping a high accuracy on clean data. We hope our approach may facilitate the development of robust applications in medical field.



## **21. Adaptive Image Transformations for Transfer-based Adversarial Attack**

cs.CV

20 pages, 6 figures, 8 tables

**SubmitDate**: 2021-11-27    [paper-pdf](http://arxiv.org/pdf/2111.13844v1)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.



## **22. Adaptive Perturbation for Adversarial Attack**

cs.CV

11 pages, 3 figures, 8 tables

**SubmitDate**: 2021-11-27    [paper-pdf](http://arxiv.org/pdf/2111.13841v1)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: In recent years, the security of deep learning models achieves more and more attentions with the rapid development of neural networks, which are vulnerable to adversarial examples. Almost all existing gradient-based attack methods use the sign function in the generation to meet the requirement of perturbation budget on $L_\infty$ norm. However, we find that the sign function may be improper for generating adversarial examples since it modifies the exact gradient direction. We propose to remove the sign function and directly utilize the exact gradient direction with a scaling factor for generating adversarial perturbations, which improves the attack success rates of adversarial examples even with fewer perturbations. Moreover, considering that the best scaling factor varies across different images, we propose an adaptive scaling factor generator to seek an appropriate scaling factor for each image, which avoids the computational cost for manually searching the scaling factor. Our method can be integrated with almost all existing gradient-based attack methods to further improve the attack success rates. Extensive experiments on the CIFAR10 and ImageNet datasets show that our method exhibits higher transferability and outperforms the state-of-the-art methods.



## **23. Benchmarking Shadow Removal for Facial Landmark Detection and Beyond**

cs.CV

**SubmitDate**: 2021-11-27    [paper-pdf](http://arxiv.org/pdf/2111.13790v1)

**Authors**: Lan Fu, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Wei Feng, Yang Liu, Song Wang

**Abstracts**: Facial landmark detection is a very fundamental and significant vision task with many important applications. In practice, facial landmark detection can be affected by a lot of natural degradations. One of the most common and important degradations is the shadow caused by light source blocking. While many advanced shadow removal methods have been proposed to recover the image quality in recent years, their effects to facial landmark detection are not well studied. For example, it remains unclear whether shadow removal could enhance the robustness of facial landmark detection to diverse shadow patterns or not. In this work, for the first attempt, we construct a novel benchmark to link two independent but related tasks (i.e., shadow removal and facial landmark detection). In particular, the proposed benchmark covers diverse face shadows with different intensities, sizes, shapes, and locations. Moreover, to mine hard shadow patterns against facial landmark detection, we propose a novel method (i.e., adversarial shadow attack), which allows us to construct a challenging subset of the benchmark for a comprehensive analysis. With the constructed benchmark, we conduct extensive analysis on three state-of-the-art shadow removal methods and three landmark detectors. The observation of this work motivates us to design a novel detection-aware shadow removal framework, which empowers shadow removal to achieve higher restoration quality and enhance the shadow robustness of deployed facial landmark detectors.



## **24. Resilient Nash Equilibrium Seeking in the Partial Information Setting**

math.OC

**SubmitDate**: 2021-11-26    [paper-pdf](http://arxiv.org/pdf/2111.13735v1)

**Authors**: Dian Gadjov, Lacra Pavel

**Abstracts**: Current research in distributed Nash equilibrium (NE) seeking in the partial information setting assumes that information is exchanged between agents that are "truthful". However, in general noncooperative games agents may consider sending misinformation to neighboring agents with the goal of further reducing their cost. Additionally, communication networks are vulnerable to attacks from agents outside the game as well as communication failures. In this paper, we propose a distributed NE seeking algorithm that is robust against adversarial agents that transmit noise, random signals, constant singles, deceitful messages, as well as being resilient to external factors such as dropped communication, jammed signals, and man in the middle attacks. The core issue that makes the problem challenging is that agents have no means of verifying if the information they receive is correct, i.e. there is no "ground truth". To address this problem, we use an observation graph, that gives truthful action information, in conjunction with a communication graph, that gives (potentially incorrect) information. By filtering information obtained from these two graphs, we show that our algorithm is resilient against adversarial agents and converges to the Nash equilibrium.



## **25. The Geometry of Adversarial Training in Binary Classification**

cs.LG

**SubmitDate**: 2021-11-26    [paper-pdf](http://arxiv.org/pdf/2111.13613v1)

**Authors**: Leon Bungert, Nicolás García Trillos, Ryan Murray

**Abstracts**: We establish an equivalence between a family of adversarial training problems for non-parametric binary classification and a family of regularized risk minimization problems where the regularizer is a nonlocal perimeter functional. The resulting regularized risk minimization problems admit exact convex relaxations of the type $L^1+$ (nonlocal) $\operatorname{TV}$, a form frequently studied in image analysis and graph-based learning. A rich geometric structure is revealed by this reformulation which in turn allows us to establish a series of properties of optimal solutions of the original problem, including the existence of minimal and maximal solutions (interpreted in a suitable sense), and the existence of regular solutions (also interpreted in a suitable sense). In addition, we highlight how the connection between adversarial training and perimeter minimization problems provides a novel, directly interpretable, statistical motivation for a family of regularized risk minimization problems involving perimeter/total variation. The majority of our theoretical results are independent of the distance used to define adversarial attacks.



## **26. Explainability-Aware One Point Attack for Point Cloud Neural Networks**

cs.CV

**SubmitDate**: 2021-11-26    [paper-pdf](http://arxiv.org/pdf/2110.04158v2)

**Authors**: Hanxiao Tan, Helena Kotthaus

**Abstracts**: With the proposition of neural networks for point clouds, deep learning has started to shine in the field of 3D object recognition while researchers have shown an increased interest to investigate the reliability of point cloud networks by adversarial attacks. However, most of the existing studies aim to deceive humans or defense algorithms, while the few that address the operation principles of the models themselves remain flawed in terms of critical point selection. In this work, we propose two adversarial methods: One Point Attack (OPA) and Critical Traversal Attack (CTA), which incorporate the explainability technologies and aim to explore the intrinsic operating principle of point cloud networks and their sensitivity against critical points perturbations. Our results show that popular point cloud networks can be deceived with almost $100\%$ success rate by shifting only one point from the input instance. In addition, we show the interesting impact of different point attribution distributions on the adversarial robustness of point cloud networks. Finally, we discuss how our approaches facilitate the explainability study for point cloud networks. To the best of our knowledge, this is the first point-cloud-based adversarial approach concerning explainability. Our code is available at https://github.com/Explain3D/Exp-One-Point-Atk-PC.



## **27. Privacy-Preserving Synthetic Smart Meters Data**

eess.SP

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2012.04475v2)

**Authors**: Ganesh Del Grosso, Georg Pichler, Pablo Piantanida

**Abstracts**: Power consumption data is very useful as it allows to optimize power grids, detect anomalies and prevent failures, on top of being useful for diverse research purposes. However, the use of power consumption data raises significant privacy concerns, as this data usually belongs to clients of a power company. As a solution, we propose a method to generate synthetic power consumption samples that faithfully imitate the originals, but are detached from the clients and their identities. Our method is based on Generative Adversarial Networks (GANs). Our contribution is twofold. First, we focus on the quality of the generated data, which is not a trivial task as no standard evaluation methods are available. Then, we study the privacy guarantees provided to members of the training set of our neural network. As a minimum requirement for privacy, we demand our neural network to be robust to membership inference attacks, as these provide a gateway for further attacks in addition to presenting a privacy threat on their own. We find that there is a compromise to be made between the privacy and the performance provided by the algorithm.



## **28. Real-Time Privacy-Preserving Data Release for Smart Meters**

eess.SP

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/1906.06427v4)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Pablo Piantanida, Fabrice Labeau

**Abstracts**: Smart Meters (SMs) are able to share the power consumption of users with utility providers almost in real-time. These fine-grained signals carry sensitive information about users, which has raised serious concerns from the privacy viewpoint. In this paper, we focus on real-time privacy threats, i.e., potential attackers that try to infer sensitive information from SMs data in an online fashion. We adopt an information-theoretic privacy measure and show that it effectively limits the performance of any attacker. Then, we propose a general formulation to design a privatization mechanism that can provide a target level of privacy by adding a minimal amount of distortion to the SMs measurements. On the other hand, to cope with different applications, a flexible distortion measure is considered. This formulation leads to a general loss function, which is optimized using a deep learning adversarial framework, where two neural networks -- referred to as the releaser and the adversary -- are trained with opposite goals. An exhaustive empirical study is then performed to validate the performance of the proposed approach and compare it with state-of-the-art methods for the occupancy detection privacy problem. Finally, we also investigate the impact of data mismatch between the releaser and the attacker.



## **29. Simple Post-Training Robustness Using Test Time Augmentations and Random Forest**

cs.CV

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2109.08191v2)

**Authors**: Gilad Cohen, Raja Giryes

**Abstracts**: Although Deep Neural Networks (DNNs) achieve excellent performance on many real-world tasks, they are highly vulnerable to adversarial attacks. A leading defense against such attacks is adversarial training, a technique in which a DNN is trained to be robust to adversarial attacks by introducing adversarial noise to its input. This procedure is effective but must be done during the training phase. In this work, we propose Augmented Random Forest (ARF), a simple and easy-to-use strategy for robustifying an existing pretrained DNN without modifying its weights. For every image, we generate randomized test time augmentations by applying diverse color, blur, noise, and geometric transforms. Then we use the DNN's logits output to train a simple random forest to predict the real class label. Our method achieves state-of-the-art adversarial robustness on a diversity of white and black box attacks with minimal compromise on the natural images' classification. We test ARF also against numerous adaptive white-box attacks and it shows excellent results when combined with adversarial training. Code is available at https://github.com/giladcohen/ARF.



## **30. EAD: an ensemble approach to detect adversarial examples from the hidden features of deep neural networks**

cs.CV

Corrected Figure 4

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12631v2)

**Authors**: Francesco Craighero, Fabrizio Angaroni, Fabio Stella, Chiara Damiani, Marco Antoniotti, Alex Graudenzi

**Abstracts**: One of the key challenges in Deep Learning is the definition of effective strategies for the detection of adversarial examples. To this end, we propose a novel approach named Ensemble Adversarial Detector (EAD) for the identification of adversarial examples, in a standard multiclass classification scenario. EAD combines multiple detectors that exploit distinct properties of the input instances in the internal representation of a pre-trained Deep Neural Network (DNN). Specifically, EAD integrates the state-of-the-art detectors based on Mahalanobis distance and on Local Intrinsic Dimensionality (LID) with a newly introduced method based on One-class Support Vector Machines (OSVMs). Although all constituting methods assume that the greater the distance of a test instance from the set of correctly classified training instances, the higher its probability to be an adversarial example, they differ in the way such distance is computed. In order to exploit the effectiveness of the different methods in capturing distinct properties of data distributions and, accordingly, efficiently tackle the trade-off between generalization and overfitting, EAD employs detector-specific distance scores as features of a logistic regression classifier, after independent hyperparameters optimization. We evaluated the EAD approach on distinct datasets (CIFAR-10, CIFAR-100 and SVHN) and models (ResNet and DenseNet) and with regard to four adversarial attacks (FGSM, BIM, DeepFool and CW), also by comparing with competing approaches. Overall, we show that EAD achieves the best AUROC and AUPR in the large majority of the settings and comparable performance in the others. The improvement over the state-of-the-art, and the possibility to easily extend EAD to include any arbitrary set of detectors, pave the way to a widespread adoption of ensemble approaches in the broad field of adversarial example detection.



## **31. Can You Spot the Chameleon? Adversarially Camouflaging Images from Co-Salient Object Detection**

cs.CV

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2009.09258v3)

**Authors**: Ruijun Gao, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Huazhu Fu, Wei Feng, Yang Liu, Song Wang

**Abstracts**: Co-salient object detection (CoSOD) has recently achieved significant progress and played a key role in retrieval-related tasks. However, it inevitably poses an entirely new safety and security issue, \ie, highly personal and sensitive content can potentially be extracting by powerful CoSOD methods. In this paper, we address this problem from the perspective of adversarial attacks and identify a novel task: adversarial co-saliency attack. Specially, given an image selected from a group of images containing some common and salient objects, we aim to generate an adversarial version that can mislead CoSOD methods to predict incorrect co-salient regions. Note that, compared with general white-box adversarial attacks for classification, this new task faces two additional challenges: (1) low success rate due to the diverse appearance of images in the group; (2) low transferability across CoSOD methods due to the considerable difference between CoSOD pipelines. To address these challenges, we propose the very first black-box joint adversarial exposure and noise attack (Jadena), where we jointly and locally tune the exposure and additive perturbations of the image according to a newly designed high-feature-level contrast-sensitive loss function. Our method, without any information on the state-of-the-art CoSOD methods, leads to significant performance degradation on various co-saliency detection datasets and makes the co-salient objects undetectable. This can have strong practical benefits in properly securing the large number of personal photos currently shared on the internet. Moreover, our method is potential to be utilized as a metric for evaluating the robustness of CoSOD methods.



## **32. AdvBokeh: Learning to Adversarially Defocus Blur**

cs.CV

13 pages

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12971v1)

**Authors**: Yihao Huang, Felix Juefei-Xu, Qing Guo, Weikai Miao, Yang Liu, Geguang Pu

**Abstracts**: Bokeh effect is a natural shallow depth-of-field phenomenon that blurs the out-of-focus part in photography. In pursuit of aesthetically pleasing photos, people usually regard the bokeh effect as an indispensable part of the photo. Due to its natural advantage and universality, as well as the fact that many visual recognition tasks can already be negatively affected by the `natural bokeh' phenomenon, in this work, we systematically study the bokeh effect from a new angle, i.e., adversarial bokeh attack (AdvBokeh) that aims to embed calculated deceptive information into the bokeh generation and produce a natural adversarial example without any human-noticeable noise artifacts. To this end, we first propose a Depth-guided Bokeh Synthesis Network (DebsNet) that is able to flexibly synthesis, refocus, and adjust the level of bokeh of the image, with a one-stage training procedure. The DebsNet allows us to tap into the bokeh generation process and attack the depth map that is needed for generating realistic bokeh (i.e., adversarially tuning the depth map) based on subsequent visual tasks. To further improve the realisticity of the adversarial bokeh, we propose depth-guided gradient-based attack to regularize the gradient.We validate the proposed method on a popular adversarial image classification dataset, i.e., NeurIPS-2017 DEV, and show that the proposed method can penetrate four state-of-the-art (SOTA) image classification networks i.e., ResNet50, VGG, DenseNet, and MobileNetV2 with a high success rate as well as high image quality. The adversarial examples obtained by AdvBokeh also exhibit high level of transferability under black-box settings. Moreover, the adversarially generated defocus blur images from the AdvBokeh can actually be capitalized to enhance the performance of SOTA defocus deblurring system, i.e., IFAN.



## **33. Normal vs. Adversarial: Salience-based Analysis of Adversarial Samples for Relation Extraction**

cs.CL

IJCKG 2021

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2104.00312v4)

**Authors**: Luoqiu Li, Xiang Chen, Zhen Bi, Xin Xie, Shumin Deng, Ningyu Zhang, Chuanqi Tan, Mosha Chen, Huajun Chen

**Abstracts**: Recent neural-based relation extraction approaches, though achieving promising improvement on benchmark datasets, have reported their vulnerability towards adversarial attacks. Thus far, efforts mostly focused on generating adversarial samples or defending adversarial attacks, but little is known about the difference between normal and adversarial samples. In this work, we take the first step to leverage the salience-based method to analyze those adversarial samples. We observe that salience tokens have a direct correlation with adversarial perturbations. We further find the adversarial perturbations are either those tokens not existing in the training set or superficial cues associated with relation labels. To some extent, our approach unveils the characters against adversarial samples. We release an open-source testbed, "DiagnoseAdv" in https://github.com/zjunlp/DiagnoseAdv.



## **34. Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks**

cs.CR

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12965v1)

**Authors**: Xiangyu Qi, Tinghao Xie, Ruizhe Pan, Jifeng Zhu, Yong Yang, Kai Bu

**Abstracts**: One major goal of the AI security community is to securely and reliably produce and deploy deep learning models for real-world applications. To this end, data poisoning based backdoor attacks on deep neural networks (DNNs) in the production stage (or training stage) and corresponding defenses are extensively explored in recent years. Ironically, backdoor attacks in the deployment stage, which can often happen in unprofessional users' devices and are thus arguably far more threatening in real-world scenarios, draw much less attention of the community. We attribute this imbalance of vigilance to the weak practicality of existing deployment-stage backdoor attack algorithms and the insufficiency of real-world attack demonstrations. To fill the blank, in this work, we study the realistic threat of deployment-stage backdoor attacks on DNNs. We base our study on a commonly used deployment-stage attack paradigm -- adversarial weight attack, where adversaries selectively modify model weights to embed backdoor into deployed DNNs. To approach realistic practicality, we propose the first gray-box and physically realizable weights attack algorithm for backdoor injection, namely subnet replacement attack (SRA), which only requires architecture information of the victim model and can support physical triggers in the real world. Extensive experimental simulations and system-level real-world attack demonstrations are conducted. Our results not only suggest the effectiveness and practicality of the proposed attack algorithm, but also reveal the practical risk of a novel type of computer virus that may widely spread and stealthily inject backdoor into DNN models in user devices. By our study, we call for more attention to the vulnerability of DNNs in the deployment stage.



## **35. Clustering Effect of (Linearized) Adversarial Robust Models**

cs.LG

Accepted by NeurIPS 2021, spotlight

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12922v1)

**Authors**: Yang Bai, Xin Yan, Yong Jiang, Shu-Tao Xia, Yisen Wang

**Abstracts**: Adversarial robustness has received increasing attention along with the study of adversarial examples. So far, existing works show that robust models not only obtain robustness against various adversarial attacks but also boost the performance in some downstream tasks. However, the underlying mechanism of adversarial robustness is still not clear. In this paper, we interpret adversarial robustness from the perspective of linear components, and find that there exist some statistical properties for comprehensively robust models. Specifically, robust models show obvious hierarchical clustering effect on their linearized sub-networks, when removing or replacing all non-linear components (e.g., batch normalization, maximum pooling, or activation layers). Based on these observations, we propose a novel understanding of adversarial robustness and apply it on more tasks including domain adaption and robustness boosting. Experimental evaluations demonstrate the rationality and superiority of our proposed clustering strategy.



## **36. Robustness against Adversarial Attacks in Neural Networks using Incremental Dissipativity**

cs.LG

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12906v1)

**Authors**: Bernardo Aquino, Arash Rahnama, Peter Seiler, Lizhen Lin, Vijay Gupta

**Abstracts**: Adversarial examples can easily degrade the classification performance in neural networks. Empirical methods for promoting robustness to such examples have been proposed, but often lack both analytical insights and formal guarantees. Recently, some robustness certificates have appeared in the literature based on system theoretic notions. This work proposes an incremental dissipativity-based robustness certificate for neural networks in the form of a linear matrix inequality for each layer. We also propose an equivalent spectral norm bound for this certificate which is scalable to neural networks with multiple layers. We demonstrate the improved performance against adversarial attacks on a feed-forward neural network trained on MNIST and an Alexnet trained using CIFAR-10.



## **37. On the Impact of Side Information on Smart Meter Privacy-Preserving Methods**

eess.SP

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2006.16062v2)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Pablo Piantanida, Fabrice Labeau

**Abstracts**: Smart meters (SMs) can pose privacy threats for consumers, an issue that has received significant attention in recent years. This paper studies the impact of Side Information (SI) on the performance of distortion-based real-time privacy-preserving algorithms for SMs. In particular, we consider a deep adversarial learning framework, in which the desired releaser (a recurrent neural network) is trained by fighting against an adversary network until convergence. To define the loss functions, two different approaches are considered: the Causal Adversarial Learning (CAL) and the Directed Information (DI)-based learning. The main difference between these approaches is in how the privacy term is measured during the training process. On the one hand, the releaser in the CAL method, by getting supervision from the actual values of the private variables and feedback from the adversary performance, tries to minimize the adversary log-likelihood. On the other hand, the releaser in the DI approach completely relies on the feedback received from the adversary and is optimized to maximize its uncertainty. The performance of these two algorithms is evaluated empirically using real-world SMs data, considering an attacker with access to SI (e.g., the day of the week) that tries to infer the occupancy status from the released SMs data. The results show that, although they perform similarly when the attacker does not exploit the SI, in general, the CAL method is less sensitive to the inclusion of SI. However, in both cases, privacy levels are significantly affected, particularly when multiple sources of SI are included.



## **38. Deep Directed Information-Based Learning for Privacy-Preserving Smart Meter Data Release**

cs.LG

to appear in IEEESmartGridComm 2019. arXiv admin note: substantial  text overlap with arXiv:1906.06427

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2011.11421v3)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Pablo Piantanida, Fabrice Labeau

**Abstracts**: The explosion of data collection has raised serious privacy concerns in users due to the possibility that sharing data may also reveal sensitive information. The main goal of a privacy-preserving mechanism is to prevent a malicious third party from inferring sensitive information while keeping the shared data useful. In this paper, we study this problem in the context of time series data and smart meters (SMs) power consumption measurements in particular. Although Mutual Information (MI) between private and released variables has been used as a common information-theoretic privacy measure, it fails to capture the causal time dependencies present in the power consumption time series data. To overcome this limitation, we introduce the Directed Information (DI) as a more meaningful measure of privacy in the considered setting and propose a novel loss function. The optimization is then performed using an adversarial framework where two Recurrent Neural Networks (RNNs), referred to as the releaser and the adversary, are trained with opposite goals. Our empirical studies on real-world data sets from SMs measurements in the worst-case scenario where an attacker has access to all the training data set used by the releaser, validate the proposed method and show the existing trade-offs between privacy and utility.



## **39. Estimating g-Leakage via Machine Learning**

cs.CR

This is the extended version of the paper which will appear in the  Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications  Security (CCS '20), November 9-13, 2020, Virtual Event, USA

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2005.04399v3)

**Authors**: Marco Romanelli, Konstantinos Chatzikokolakis, Catuscia Palamidessi, Pablo Piantanida

**Abstracts**: This paper considers the problem of estimating the information leakage of a system in the black-box scenario. It is assumed that the system's internals are unknown to the learner, or anyway too complicated to analyze, and the only available information are pairs of input-output data samples, possibly obtained by submitting queries to the system or provided by a third party. Previous research has mainly focused on counting the frequencies to estimate the input-output conditional probabilities (referred to as frequentist approach), however this method is not accurate when the domain of possible outputs is large. To overcome this difficulty, the estimation of the Bayes error of the ideal classifier was recently investigated using Machine Learning (ML) models and it has been shown to be more accurate thanks to the ability of those models to learn the input-output correspondence. However, the Bayes vulnerability is only suitable to describe one-try attacks. A more general and flexible measure of leakage is the g-vulnerability, which encompasses several different types of adversaries, with different goals and capabilities. In this paper, we propose a novel approach to perform black-box estimation of the g-vulnerability using ML. A feature of our approach is that it does not require to estimate the conditional probabilities, and that it is suitable for a large class of ML algorithms. First, we formally show the learnability for all data distributions. Then, we evaluate the performance via various experiments using k-Nearest Neighbors and Neural Networks. Our results outperform the frequentist approach when the observables domain is large.



## **40. SoK: Plausibly Deniable Storage**

cs.CR

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.12809v1)

**Authors**: Chen Chen, Xiao Liang, Bogdan Carbunar, Radu Sion

**Abstracts**: Data privacy is critical in instilling trust and empowering the societal pacts of modern technology-driven democracies. Unfortunately, it is under continuous attack by overreaching or outright oppressive governments, including some of the world's oldest democracies. Increasingly-intrusive anti-encryption laws severely limit the ability of standard encryption to protect privacy. New defense mechanisms are needed.   Plausible deniability (PD) is a powerful property, enabling users to hide the existence of sensitive information in a system under direct inspection by adversaries. Popular encrypted storage systems such as TrueCrypt and other research efforts have attempted to also provide plausible deniability. Unfortunately, these efforts have often operated under less well-defined assumptions and adversarial models. Careful analyses often uncover not only high overheads but also outright security compromise. Further, our understanding of adversaries, the underlying storage technologies, as well as the available plausible deniable solutions have evolved dramatically in the past two decades. The main goal of this work is to systematize this knowledge. It aims to:   - identify key PD properties, requirements, and approaches;   - present a direly-needed unified framework for evaluating security and performance;   - explore the challenges arising from the critical interplay between PD and modern system layered stacks;   - propose a new "trace-oriented" PD paradigm, able to decouple security guarantees from the underlying systems and thus ensure a higher level of flexibility and security independent of the technology stack.   This work is meant also as a trusted guide for system and security practitioners around the major challenges in understanding, designing, and implementing plausible deniability into new or existing systems.



## **41. On the Effect of Pruning on Adversarial Robustness**

cs.CV

Published at International Conference on Computer Vision Workshop  (ICCVW), 2021

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2108.04890v2)

**Authors**: Artur Jordao, Helio Pedrini

**Abstracts**: Pruning is a well-known mechanism for reducing the computational cost of deep convolutional networks. However, studies have shown the potential of pruning as a form of regularization, which reduces overfitting and improves generalization. We demonstrate that this family of strategies provides additional benefits beyond computational performance and generalization. Our analyses reveal that pruning structures (filters and/or layers) from convolutional networks increase not only generalization but also robustness to adversarial images (natural images with content modified). Such achievements are possible since pruning reduces network capacity and provides regularization, which have been proven effective tools against adversarial images. In contrast to promising defense mechanisms that require training with adversarial images and careful regularization, we show that pruning obtains competitive results considering only natural images (e.g., the standard and low-cost training). We confirm these findings on several adversarial attacks and architectures; thus suggesting the potential of pruning as a novel defense mechanism against adversarial images.



## **42. Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses**

cs.LG

13 pages, 6 figures

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2106.08746v2)

**Authors**: Buse G. A. Tekgul, Shelly Wang, Samuel Marchal, N. Asokan

**Abstracts**: Recent work has shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial perturbations. Adversaries can mislead policies of DRL agents by perturbing the state of the environment observed by the agents. Existing attacks are feasible in principle but face challenges in practice, for example by being too slow to fool DRL policies in real time. We show that using the Universal Adversarial Perturbation (UAP) method to compute perturbations, independent of the individual inputs to which they are applied to, can fool DRL policies effectively and in real time. We describe three such attack variants. Via an extensive evaluation using three Atari 2600 games, we show that our attacks are effective, as they fully degrade the performance of three different DRL agents (up to 100%, even when the $l_\infty$ bound on the perturbation is as small as 0.01). It is faster compared to the response time (0.6ms on average) of different DRL policies, and considerably faster than prior attacks using adversarial perturbations (1.8ms on average). We also show that our attack technique is efficient, incurring an online computational cost of 0.027ms on average. Using two further tasks involving robotic movement, we confirm that our results generalize to more complex DRL tasks. Furthermore, we demonstrate that the effectiveness of known defenses diminishes against universal perturbations. We propose an effective technique that detects all known adversarial perturbations against DRL policies, including all the universal perturbations presented in this paper.



## **43. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

cs.LG

22 pages, 15 figures, 5 tables

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.06628v2)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.



## **44. REGroup: Rank-aggregating Ensemble of Generative Classifiers for Robust Predictions**

cs.CV

WACV,2022. Project Page : https://lokender.github.io/REGroup.html

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2006.10679v2)

**Authors**: Lokender Tiwari, Anish Madan, Saket Anand, Subhashis Banerjee

**Abstracts**: Deep Neural Networks (DNNs) are often criticized for being susceptible to adversarial attacks. Most successful defense strategies adopt adversarial training or random input transformations that typically require retraining or fine-tuning the model to achieve reasonable performance. In this work, our investigations of intermediate representations of a pre-trained DNN lead to an interesting discovery pointing to intrinsic robustness to adversarial attacks. We find that we can learn a generative classifier by statistically characterizing the neural response of an intermediate layer to clean training samples. The predictions of multiple such intermediate-layer based classifiers, when aggregated, show unexpected robustness to adversarial attacks. Specifically, we devise an ensemble of these generative classifiers that rank-aggregates their predictions via a Borda count-based consensus. Our proposed approach uses a subset of the clean training data and a pre-trained model, and yet is agnostic to network architectures or the adversarial attack generation method. We show extensive experiments to establish that our defense strategy achieves state-of-the-art performance on the ImageNet validation set.



## **45. Thundernna: a white box adversarial attack**

cs.LG

10 pages, 5 figures

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.12305v1)

**Authors**: Linfeng Ye

**Abstracts**: The existing work shows that the neural network trained by naive gradient-based optimization method is prone to adversarial attacks, adds small malicious on the ordinary input is enough to make the neural network wrong. At the same time, the attack against a neural network is the key to improving its robustness. The training against adversarial examples can make neural networks resist some kinds of adversarial attacks. At the same time, the adversarial attack against a neural network can also reveal some characteristics of the neural network, a complex high-dimensional non-linear function, as discussed in previous work.   In This project, we develop a first-order method to attack the neural network. Compare with other first-order attacks, our method has a much higher success rate. Furthermore, it is much faster than second-order attacks and multi-steps first-order attacks.



## **46. Subspace Adversarial Training**

cs.LG

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.12229v1)

**Authors**: Tao Li, Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: Single-step adversarial training (AT) has received wide attention as it proved to be both efficient and robust. However, a serious problem of catastrophic overfitting exists, i.e., the robust accuracy against projected gradient descent (PGD) attack suddenly drops to $0\%$ during the training. In this paper, we understand this problem from a novel perspective of optimization and firstly reveal the close link between the fast-growing gradient of each sample and overfitting, which can also be applied to understand the robust overfitting phenomenon in multi-step AT. To control the growth of the gradient during the training, we propose a new AT method, subspace adversarial training (Sub-AT), which constrains the AT in a carefully extracted subspace. It successfully resolves both two kinds of overfitting and hence significantly boosts the robustness. In subspace, we also allow single-step AT with larger steps and larger radius, which further improves the robustness performance. As a result, we achieve the state-of-the-art single-step AT performance: our pure single-step AT can reach over $\mathbf{51}\%$ robust accuracy against strong PGD-50 attack with radius $8/255$ on CIFAR-10, even surpassing the standard multi-step PGD-10 AT with huge computational advantages. The code is released$\footnote{\url{https://github.com/nblt/Sub-AT}}$.



## **47. Fixed Points in Cyber Space: Rethinking Optimal Evasion Attacks in the Age of AI-NIDS**

cs.CR

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2111.12197v1)

**Authors**: Christian Schroeder de Witt, Yongchao Huang, Philip H. S. Torr, Martin Strohmeier

**Abstracts**: Cyber attacks are increasing in volume, frequency, and complexity. In response, the security community is looking toward fully automating cyber defense systems using machine learning. However, so far the resultant effects on the coevolutionary dynamics of attackers and defenders have not been examined. In this whitepaper, we hypothesise that increased automation on both sides will accelerate the coevolutionary cycle, thus begging the question of whether there are any resultant fixed points, and how they are characterised. Working within the threat model of Locked Shields, Europe's largest cyberdefense exercise, we study blackbox adversarial attacks on network classifiers. Given already existing attack capabilities, we question the utility of optimal evasion attack frameworks based on minimal evasion distances. Instead, we suggest a novel reinforcement learning setting that can be used to efficiently generate arbitrary adversarial perturbations. We then argue that attacker-defender fixed points are themselves general-sum games with complex phase transitions, and introduce a temporally extended multi-agent reinforcement learning framework in which the resultant dynamics can be studied. We hypothesise that one plausible fixed point of AI-NIDS may be a scenario where the defense strategy relies heavily on whitelisted feature flow subspaces. Finally, we demonstrate that a continual learning approach is required to study attacker-defender dynamics in temporally extended general-sum games.



## **48. Watermarking Graph Neural Networks based on Backdoor Attacks**

cs.LG

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2110.11024v2)

**Authors**: Jing Xu, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise on fine-tuning the model. What is more, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, it is necessary to verify the ownership of the GNN models.   In this paper, we present a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (around $100\%$) for both tasks. In addition, we experimentally show that our watermarking approach is still effective even when considering suspicious models obtained from different architectures than the owner's.



## **49. Adversarial machine learning for protecting against online manipulation**

cs.LG

To appear on IEEE Internet Computing. `Accepted manuscript' version

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2111.12034v1)

**Authors**: Stefano Cresci, Marinella Petrocchi, Angelo Spognardi, Stefano Tognazzi

**Abstracts**: Adversarial examples are inputs to a machine learning system that result in an incorrect output from that system. Attacks launched through this type of input can cause severe consequences: for example, in the field of image recognition, a stop signal can be misclassified as a speed limit indication.However, adversarial examples also represent the fuel for a flurry of research directions in different domains and applications. Here, we give an overview of how they can be profitably exploited as powerful tools to build stronger learning models, capable of better-withstanding attacks, for two crucial tasks: fake news and social bot detection.



## **50. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

cs.NE

14 pages, 7 figures, 4 tables and 20 References

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2110.01818v3)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the search ability, convergence efficiency and precision of genetic algorithms. In this paper, a new improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by four test functions. Simulation results show that, comparing with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. Finally, the algorithm is applied to neural networks adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.



