# New Adversarial Attack Papers
**update at 2021-11-18 23:56:42**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Cortical Features for Defense Against Adversarial Audio Attacks**

cs.SD

Co-author legal name changed

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2102.00313v2)

**Authors**: Ilya Kavalerov, Ruijie Zheng, Wojciech Czaja, Rama Chellappa

**Abstracts**: We propose using a computational model of the auditory cortex as a defense against adversarial attacks on audio. We apply several white-box iterative optimization-based adversarial attacks to an implementation of Amazon Alexa's HW network, and a modified version of this network with an integrated cortical representation, and show that the cortical features help defend against universal adversarial examples. At the same level of distortion, the adversarial noises found for the cortical network are always less effective for universal audio attacks. We make our code publicly available at https://github.com/ilyakava/py3fst.



## **2. Address Behaviour Vulnerabilities in the Next Generation of Autonomous Robots**

cs.RO

preprint and extended version of Nature Machine Intelligence, Vol 3,  November 2021, Pag 927-928

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2103.13268v2)

**Authors**: Michele Colledanchise

**Abstracts**: Robots applications in our daily life increase at an unprecedented pace. As robots will soon operate "out in the wild", we must identify the safety and security vulnerabilities they will face. Robotics researchers and manufacturers focus their attention on new, cheaper, and more reliable applications. Still, they often disregard the operability in adversarial environments where a trusted or untrusted user can jeopardize or even alter the robot's task.   In this paper, we identify a new paradigm of security threats in the next generation of robots. These threats fall beyond the known hardware or network-based ones, and we must find new solutions to address them. These new threats include malicious use of the robot's privileged access, tampering with the robot sensors system, and tricking the robot's deliberation into harmful behaviors. We provide a taxonomy of attacks that exploit these vulnerabilities with realistic examples, and we outline effective countermeasures to prevent better, detect, and mitigate them.



## **3. Do Not Trust Prediction Scores for Membership Inference Attacks**

cs.LG

15 pages, 9 figures, 9 tables

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.09076v1)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstracts**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Arguably, most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures, e.g., ReLU type neural networks produce almost always high prediction scores far away from the training data. Consequently, MIAs will miserably fail since this behavior leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of classifiers and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions far away from the training data, the more they reveal the training data.



## **4. Generating Unrestricted 3D Adversarial Point Clouds**

cs.CV

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.08973v1)

**Authors**: Xuelong Dai, Yanjie Li, Hua Dai, Bin Xiao

**Abstracts**: Utilizing 3D point cloud data has become an urgent need for the deployment of artificial intelligence in many areas like facial recognition and self-driving. However, deep learning for 3D point clouds is still vulnerable to adversarial attacks, e.g., iterative attacks, point transformation attacks, and generative attacks. These attacks need to restrict perturbations of adversarial examples within a strict bound, leading to the unrealistic adversarial 3D point clouds. In this paper, we propose an Adversarial Graph-Convolutional Generative Adversarial Network (AdvGCGAN) to generate visually realistic adversarial 3D point clouds from scratch. Specifically, we use a graph convolutional generator and a discriminator with an auxiliary classifier to generate realistic point clouds, which learn the latent distribution from the real 3D data. The unrestricted adversarial attack loss is incorporated in the special adversarial training of GAN, which enables the generator to generate the adversarial examples to spoof the target network. Compared with the existing state-of-art attack methods, the experiment results demonstrate the effectiveness of our unrestricted adversarial attack methods with a higher attack success rate and visual quality. Additionally, the proposed AdvGCGAN can achieve better performance against defense models and better transferability than existing attack methods with strong camouflage.



## **5. TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**

cs.CV

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.08954v1)

**Authors**: Delv Lin, Qi Chen, Chengyu Zhou, Kun He

**Abstracts**: Benefiting from the development of Deep Neural Networks, Multi-Object Tracking (MOT) has achieved aggressive progress. Currently, the real-time Joint-Detection-Tracking (JDT) based MOT trackers gain increasing attention and derive many excellent models. However, the robustness of JDT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during tracking. In this work, we analyze the weakness of JDT trackers and propose a novel adversarial attack method, called Tracklet-Switch (TraSw), against the complete tracking pipeline of MOT. Specifically, a push-pull loss and a center leaping optimization are designed to generate adversarial examples for both re-ID feature and object detection. TraSw can fool the tracker to fail to track the targets in the subsequent frames by attacking very few frames. We evaluate our method on the advanced deep trackers (i.e., FairMOT, JDE, ByteTrack) using the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20). Experiments show that TraSw can achieve a high success rate of over 95% by attacking only five frames on average for the single-target attack and a reasonably high success rate of over 80% for the multiple-target attack. The code is available at https://github.com/DerryHub/FairMOT-attack .



## **6. Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attacks**

cs.CR

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2108.05075v2)

**Authors**: Zitao Chen, Pritam Dash, Karthik Pattabiraman

**Abstracts**: Adversarial patch attacks against image classification deep neural networks (DNNs), which inject arbitrary distortions within a bounded region of an image, can generate adversarial perturbations that are robust (i.e., remain adversarial in physical world) and universal (i.e., remain adversarial on any input). Such attacks can lead to severe consequences in real-world DNN-based systems.   This work proposes Jujutsu, a technique to detect and mitigate robust and universal adversarial patch attacks. For detection, Jujutsu exploits the attacks' universal property - Jujutsu first locates the region of the potential adversarial patch, and then strategically transfers it to a dedicated region in a new image to determine whether it is truly malicious. For attack mitigation, Jujutsu leverages the attacks' localized nature via image inpainting to synthesize the semantic contents in the pixels that are corrupted by the attacks, and reconstruct the ``clean'' image.   We evaluate Jujutsu on four diverse datasets (ImageNet, ImageNette, CelebA and Place365), and show that Jujutsu achieves superior performance and significantly outperforms existing techniques. We find that Jujutsu can further defend against different variants of the basic attack, including 1) physical-world attack; 2) attacks that target diverse classes; 3) attacks that construct patches in different shapes and 4) adaptive attacks.



## **7. Detecting AutoAttack Perturbations in the Frequency Domain**

cs.CV

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08785v1)

**Authors**: Peter Lorenz, Paula Harder, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstracts**: Recently, adversarial attacks on image classification networks by the AutoAttack (Croce and Hein, 2020b) framework have drawn a lot of attention. While AutoAttack has shown a very high attack success rate, most defense approaches are focusing on network hardening and robustness enhancements, like adversarial training. This way, the currently best-reported method can withstand about 66% of adversarial examples on CIFAR10. In this paper, we investigate the spatial and frequency domain properties of AutoAttack and propose an alternative defense. Instead of hardening a network, we detect adversarial attacks during inference, rejecting manipulated inputs. Based on a rather simple and fast analysis in the frequency domain, we introduce two different detection algorithms. First, a black box detector that only operates on the input images and achieves a detection accuracy of 100% on the AutoAttack CIFAR10 benchmark and 99.3% on ImageNet, for epsilon = 8/255 in both cases. Second, a whitebox detector using an analysis of CNN feature maps, leading to a detection rate of also 100% and 98.7% on the same benchmarks.



## **8. Robustness of Bayesian Neural Networks to White-Box Adversarial Attacks**

cs.LG

Accepted at the fourth IEEE International Conference on Artificial  Intelligence and Knowledge Engineering (AIKE 2021)

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08591v1)

**Authors**: Adaku Uchendu, Daniel Campoy, Christopher Menart, Alexandra Hildenbrandt

**Abstracts**: Bayesian Neural Networks (BNNs), unlike Traditional Neural Networks (TNNs) are robust and adept at handling adversarial attacks by incorporating randomness. This randomness improves the estimation of uncertainty, a feature lacking in TNNs. Thus, we investigate the robustness of BNNs to white-box attacks using multiple Bayesian neural architectures. Furthermore, we create our BNN model, called BNN-DenseNet, by fusing Bayesian inference (i.e., variational Bayes) to the DenseNet architecture, and BDAV, by combining this intervention with adversarial training. Experiments are conducted on the CIFAR-10 and FGVC-Aircraft datasets. We attack our models with strong white-box attacks ($l_\infty$-FGSM, $l_\infty$-PGD, $l_2$-PGD, EOT $l_\infty$-FGSM, and EOT $l_\infty$-PGD). In all experiments, at least one BNN outperforms traditional neural networks during adversarial attack scenarios. An adversarially-trained BNN outperforms its non-Bayesian, adversarially-trained counterpart in most experiments, and often by significant margins. Lastly, we investigate network calibration and find that BNNs do not make overconfident predictions, providing evidence that BNNs are also better at measuring uncertainty.



## **9. Improving the robustness and accuracy of biomedical language models through adversarial training**

cs.CL

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08529v1)

**Authors**: Milad Moradi, Matthias Samwald

**Abstracts**: Deep transformer neural network models have improved the predictive accuracy of intelligent text processing systems in the biomedical domain. They have obtained state-of-the-art performance scores on a wide variety of biomedical and clinical Natural Language Processing (NLP) benchmarks. However, the robustness and reliability of these models has been less explored so far. Neural NLP models can be easily fooled by adversarial samples, i.e. minor changes to input that preserve the meaning and understandability of the text but force the NLP system to make erroneous decisions. This raises serious concerns about the security and trust-worthiness of biomedical NLP systems, especially when they are intended to be deployed in real-world use cases. We investigated the robustness of several transformer neural language models, i.e. BioBERT, SciBERT, BioMed-RoBERTa, and Bio-ClinicalBERT, on a wide range of biomedical and clinical text processing tasks. We implemented various adversarial attack methods to test the NLP systems in different attack scenarios. Experimental results showed that the biomedical NLP models are sensitive to adversarial samples; their performance dropped in average by 21 and 18.9 absolute percent on character-level and word-level adversarial noise, respectively. Conducting extensive adversarial training experiments, we fine-tuned the NLP models on a mixture of clean samples and adversarial inputs. Results showed that adversarial training is an effective defense mechanism against adversarial noise; the models robustness improved in average by 11.3 absolute percent. In addition, the models performance on clean data increased in average by 2.4 absolute present, demonstrating that adversarial training can boost generalization abilities of biomedical NLP systems.



## **10. Consistent Semantic Attacks on Optical Flow**

cs.CV

Paper and supplementary material

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08485v1)

**Authors**: Tom Koren, Lior Talker, Michael Dinerstein, Roy J Jevnisek

**Abstracts**: We present a novel approach for semantically targeted adversarial attacks on Optical Flow. In such attacks the goal is to corrupt the flow predictions of a specific object category or instance. Usually, an attacker seeks to hide the adversarial perturbations in the input. However, a quick scan of the output reveals the attack. In contrast, our method helps to hide the attackers intent in the output as well. We achieve this thanks to a regularization term that encourages off-target consistency. We perform extensive tests on leading optical flow models to demonstrate the benefits of our approach in both white-box and black-box settings. Also, we demonstrate the effectiveness of our attack on subsequent tasks that depend on the optical flow.



## **11. Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks**

cs.LG

accepted at NeurIPS 2021; updated the numbers in Table 5 and added  references

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.01714v2)

**Authors**: Maksym Yatsura, Jan Hendrik Metzen, Matthias Hein

**Abstracts**: Adversarial attacks based on randomized search schemes have obtained state-of-the-art results in black-box robustness evaluation recently. However, as we demonstrate in this work, their efficiency in different query budget regimes depends on manual design and heuristic tuning of the underlying proposal distributions. We study how this issue can be addressed by adapting the proposal distribution online based on the information obtained during the attack. We consider Square Attack, which is a state-of-the-art score-based black-box attack, and demonstrate how its performance can be improved by a learned controller that adjusts the parameters of the proposal distribution online during the attack. We train the controller using gradient-based end-to-end training on a CIFAR10 model with white box access. We demonstrate that plugging the learned controller into the attack consistently improves its black-box robustness estimate in different query regimes by up to 20% for a wide range of different models with black-box access. We further show that the learned adaptation principle transfers well to the other data distributions such as CIFAR100 or ImageNet and to the targeted attack setting.



## **12. Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework**

cs.CL

Work on progress

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2110.15317v2)

**Authors**: Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei

**Abstracts**: Despite great success on many machine learning tasks, deep neural networks are still vulnerable to adversarial samples. While gradient-based adversarial attack methods are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of text. To bridge this gap, we propose a general framework to adapt existing gradient-based methods to craft textual adversarial samples. In this framework, gradient-based continuous perturbations are added to the embedding layer and are amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a mask language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with \textbf{T}extual \textbf{P}rojected \textbf{G}radient \textbf{D}escent (\textbf{TPGD}). We conduct comprehensive experiments to evaluate our framework by performing transfer black-box attacks on BERT, RoBERTa and ALBERT on three benchmark datasets. Experimental results demonstrate our method achieves an overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. All the code and data will be made public.



## **13. InFlow: Robust outlier detection utilizing Normalizing Flows**

cs.LG

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2106.12894v2)

**Authors**: Nishant Kumar, Pia Hanfeld, Michael Hecht, Michael Bussmann, Stefan Gumhold, Nico Hoffmann

**Abstracts**: Normalizing flows are prominent deep generative models that provide tractable probability distributions and efficient density estimation. However, they are well known to fail while detecting Out-of-Distribution (OOD) inputs as they directly encode the local features of the input representations in their latent space. In this paper, we solve this overconfidence issue of normalizing flows by demonstrating that flows, if extended by an attention mechanism, can reliably detect outliers including adversarial attacks. Our approach does not require outlier data for training and we showcase the efficiency of our method for OOD detection by reporting state-of-the-art performance in diverse experimental settings. Code available at https://github.com/ComputationalRadiationPhysics/InFlow .



## **14. TSS: Transformation-Specific Smoothing for Robustness Certification**

cs.LG

2021 ACM SIGSAC Conference on Computer and Communications Security  (CCS '21)

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2002.12398v5)

**Authors**: Linyi Li, Maurice Weber, Xiaojun Xu, Luka Rimanic, Bhavya Kailkhura, Tao Xie, Ce Zhang, Bo Li

**Abstracts**: As machine learning (ML) systems become pervasive, safeguarding their security is critical. However, recently it has been demonstrated that motivated adversaries are able to mislead ML systems by perturbing test data using semantic transformations. While there exists a rich body of research providing provable robustness guarantees for ML models against $\ell_p$ norm bounded adversarial perturbations, guarantees against semantic perturbations remain largely underexplored. In this paper, we provide TSS -- a unified framework for certifying ML robustness against general adversarial semantic transformations. First, depending on the properties of each transformation, we divide common transformations into two categories, namely resolvable (e.g., Gaussian blur) and differentially resolvable (e.g., rotation) transformations. For the former, we propose transformation-specific randomized smoothing strategies and obtain strong robustness certification. The latter category covers transformations that involve interpolation errors, and we propose a novel approach based on stratified sampling to certify the robustness. Our framework TSS leverages these certification strategies and combines with consistency-enhanced training to provide rigorous certification of robustness. We conduct extensive experiments on over ten types of challenging semantic transformations and show that TSS significantly outperforms the state of the art. Moreover, to the best of our knowledge, TSS is the first approach that achieves nontrivial certified robustness on the large-scale ImageNet dataset. For instance, our framework achieves 30.4% certified robust accuracy against rotation attack (within $\pm 30^\circ$) on ImageNet. Moreover, to consider a broader range of transformations, we show TSS is also robust against adaptive attacks and unforeseen image corruptions such as CIFAR-10-C and ImageNet-C.



## **15. Android HIV: A Study of Repackaging Malware for Evading Machine-Learning Detection**

cs.CR

14 pages, 11 figures

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/1808.04218v4)

**Authors**: Xiao Chen, Chaoran Li, Derui Wang, Sheng Wen, Jun Zhang, Surya Nepal, Yang Xiang, Kui Ren

**Abstracts**: Machine learning based solutions have been successfully employed for automatic detection of malware on Android. However, machine learning models lack robustness to adversarial examples, which are crafted by adding carefully chosen perturbations to the normal inputs. So far, the adversarial examples can only deceive detectors that rely on syntactic features (e.g., requested permissions, API calls, etc), and the perturbations can only be implemented by simply modifying application's manifest. While recent Android malware detectors rely more on semantic features from Dalvik bytecode rather than manifest, existing attacking/defending methods are no longer effective. In this paper, we introduce a new attacking method that generates adversarial examples of Android malware and evades being detected by the current models. To this end, we propose a method of applying optimal perturbations onto Android APK that can successfully deceive the machine learning detectors. We develop an automated tool to generate the adversarial examples without human intervention. In contrast to existing works, the adversarial examples crafted by our method can also deceive recent machine learning based detectors that rely on semantic features such as control-flow-graph. The perturbations can also be implemented directly onto APK's Dalvik bytecode rather than Android manifest to evade from recent detectors. We demonstrate our attack on two state-of-the-art Android malware detection schemes, MaMaDroid and Drebin. Our results show that the malware detection rates decreased from 96% to 0% in MaMaDroid, and from 97% to 0% in Drebin, with just a small number of codes to be inserted into the APK.



## **16. A Survey on Adversarial Attacks for Malware Analysis**

cs.CR

42 Pages, 31 Figures, 11 Tables

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08223v1)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam

**Abstracts**: Machine learning has witnessed tremendous growth in its adoption and advancement in the last decade. The evolution of machine learning from traditional algorithms to modern deep learning architectures has shaped the way today's technology functions. Its unprecedented ability to discover knowledge/patterns from unstructured data and automate the decision-making process led to its application in wide domains. High flying machine learning arena has been recently pegged back by the introduction of adversarial attacks. Adversaries are able to modify data, maximizing the classification error of the models. The discovery of blind spots in machine learning models has been exploited by adversarial attackers by generating subtle intentional perturbations in test samples. Increasing dependency on data has paved the blueprint for ever-high incentives to camouflage machine learning models. To cope with probable catastrophic consequences in the future, continuous research is required to find vulnerabilities in form of adversarial and design remedies in systems. This survey aims at providing the encyclopedic introduction to adversarial attacks that are carried out against malware detection systems. The paper will introduce various machine learning techniques used to generate adversarial and explain the structure of target files. The survey will also model the threat posed by the adversary and followed by brief descriptions of widely accepted adversarial algorithms. Work will provide a taxonomy of adversarial evasion attacks on the basis of attack domain and adversarial generation techniques. Adversarial evasion attacks carried out against malware detectors will be discussed briefly under each taxonomical headings and compared with concomitant researches. Analyzing the current research challenges in an adversarial generation, the survey will conclude by pinpointing the open future research directions.



## **17. FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning**

cs.LG

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08211v1)

**Authors**: Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Qiang Yang

**Abstracts**: Federated learning (FL) aims to protect data privacy by enabling clients to collaboratively build machine learning models without sharing their private data. However, recent works demonstrate that FL is vulnerable to gradient-based data recovery attacks. Varieties of privacy-preserving technologies have been leveraged to further enhance the privacy of FL. Nonetheless, they either are computational or communication expensive (e.g., homomorphic encryption) or suffer from precision loss (e.g., differential privacy). In this work, we propose \textsc{FedCG}, a novel \underline{fed}erated learning method that leverages \underline{c}onditional \underline{g}enerative adversarial networks to achieve high-level privacy protection while still maintaining competitive model performance. More specifically, \textsc{FedCG} decomposes each client's local network into a private extractor and a public classifier and keeps the extractor local to protect privacy. Instead of exposing extractors which is the culprit of privacy leakage, \textsc{FedCG} shares clients' generators with the server for aggregating common knowledge aiming to enhance the performance of clients' local networks. Extensive experiments demonstrate that \textsc{FedCG} can achieve competitive model performance compared with baseline FL methods, and numerical privacy analysis shows that \textsc{FedCG} has high-level privacy-preserving capability.



## **18. 3D Adversarial Attacks Beyond Point Cloud**

cs.CV

8 pages, 6 figs

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2104.12146v3)

**Authors**: Jinlai Zhang, Lyujie Chen, Binbin Liu, Bo Ouyang, Qizhi Xie, Jihong Zhu, Weiming Li, Yanmei Meng

**Abstracts**: Recently, 3D deep learning models have been shown to be susceptible to adversarial attacks like their 2D counterparts. Most of the state-of-the-art (SOTA) 3D adversarial attacks perform perturbation to 3D point clouds. To reproduce these attacks in the physical scenario, a generated adversarial 3D point cloud need to be reconstructed to mesh, which leads to a significant drop in its adversarial effect. In this paper, we propose a strong 3D adversarial attack named Mesh Attack to address this problem by directly performing perturbation on mesh of a 3D object. In order to take advantage of the most effective gradient-based attack, a differentiable sample module that back-propagate the gradient of point cloud to mesh is introduced. To further ensure the adversarial mesh examples without outlier and 3D printable, three mesh losses are adopted. Extensive experiments demonstrate that the proposed scheme outperforms SOTA 3D attacks by a significant margin. We also achieved SOTA performance under various defenses. Our code is available at: https://github.com/cuge1995/Mesh-Attack.



## **19. Augmenting Zero Trust Architecture to Endpoints Using Blockchain: A State-of-The-Art Review**

cs.CR

(1) Fixed the reference numbering (2) Fixed syntax errors,  improvements (3) document re-structured

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2104.00460v4)

**Authors**: Lampis Alevizos, Vinh Thong Ta, Max Hashem Eiza

**Abstracts**: With the purpose of defending against lateral movement in today's borderless networks, Zero Trust Architecture (ZTA) adoption is gaining momentum. With a full scale ZTA implementation, it is unlikely that adversaries will be able to spread through the network starting from a compromised endpoint. However, the already authenticated and authorised session of a compromised endpoint can be leveraged to perform limited, though malicious, activities ultimately rendering the endpoints the Achilles heel of ZTA. To effectively detect such attacks, distributed collaborative intrusion detection systems with an attack scenario-based approach have been developed. Nonetheless, Advanced Persistent Threats (APTs) have demonstrated their ability to bypass this approach with a high success ratio. As a result, adversaries can pass undetected or potentially alter the detection logging mechanisms to achieve a stealthy presence. Recently, blockchain technology has demonstrated solid use cases in the cyber security domain. In this paper, motivated by the convergence of ZTA and blockchain-based intrusion detection and prevention, we examine how ZTA can be augmented onto endpoints. Namely, we perform a state-of-the-art review of ZTA models, real-world architectures with a focus on endpoints, and blockchain-based intrusion detection systems. We discuss the potential of blockchain's immutability fortifying the detection process and identify open challenges as well as potential solutions and future directions.



## **20. NNoculation: Catching BadNets in the Wild**

cs.CR

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2002.08313v2)

**Authors**: Akshaj Kumar Veldanda, Kang Liu, Benjamin Tan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Brendan Dolan-Gavitt, Siddharth Garg

**Abstracts**: This paper proposes a novel two-stage defense (NNoculation) against backdoored neural networks (BadNets) that, repairs a BadNet both pre-deployment and online in response to backdoored test inputs encountered in the field. In the pre-deployment stage, NNoculation retrains the BadNet with random perturbations of clean validation inputs to partially reduce the adversarial impact of a backdoor. Post-deployment, NNoculation detects and quarantines backdoored test inputs by recording disagreements between the original and pre-deployment patched networks. A CycleGAN is then trained to learn transformations between clean validation and quarantined inputs; i.e., it learns to add triggers to clean validation images. Backdoored validation images along with their correct labels are used to further retrain the pre-deployment patched network, yielding our final defense. Empirical evaluation on a comprehensive suite of backdoor attacks show that NNoculation outperforms all state-of-the-art defenses that make restrictive assumptions and only work on specific backdoor attacks, or fail on adaptive attacks. In contrast, NNoculation makes minimal assumptions and provides an effective defense, even under settings where existing defenses are ineffective due to attackers circumventing their restrictive assumptions.



## **21. Generative Dynamic Patch Attack**

cs.CV

Published as a conference paper at BMVC 2021

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.04266v2)

**Authors**: Xiang Li, Shihao Ji

**Abstracts**: Adversarial patch attack is a family of attack algorithms that perturb a part of image to fool a deep neural network model. Existing patch attacks mostly consider injecting adversarial patches at input-agnostic locations: either a predefined location or a random location. This attack setup may be sufficient for attack but has considerable limitations when using it for adversarial training. Thus, robust models trained with existing patch attacks cannot effectively defend other adversarial attacks. In this paper, we first propose an end-to-end patch attack algorithm, Generative Dynamic Patch Attack (GDPA), which generates both patch pattern and patch location adversarially for each input image. We show that GDPA is a generic attack framework that can produce dynamic/static and visible/invisible patches with a few configuration changes. Secondly, GDPA can be readily integrated for adversarial training to improve model robustness to various adversarial attacks. Extensive experiments on VGGFace, Traffic Sign and ImageNet show that GDPA achieves higher attack success rates than state-of-the-art patch attacks, while adversarially trained model with GDPA demonstrates superior robustness to adversarial patch attacks than competing methods. Our source code can be found at https://github.com/lxuniverse/gdpa.



## **22. Website fingerprinting on early QUIC traffic**

cs.CR

This work has been accepted by Elsevier Computer Networks for  publication

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2101.11871v2)

**Authors**: Pengwei Zhan, Liming Wang, Yi Tang

**Abstracts**: Cryptographic protocols have been widely used to protect the user's privacy and avoid exposing private information. QUIC (Quick UDP Internet Connections), including the version originally designed by Google (GQUIC) and the version standardized by IETF (IQUIC), as alternatives to the traditional HTTP, demonstrate their unique transmission characteristics: based on UDP for encrypted resource transmitting, accelerating web page rendering. However, existing encrypted transmission schemes based on TCP are vulnerable to website fingerprinting (WFP) attacks, allowing adversaries to infer the users' visited websites by eavesdropping on the transmission channel. Whether GQUIC and IQUIC can effectively resist such attacks is worth investigating. In this paper, we study the vulnerabilities of GQUIC, IQUIC, and HTTPS to WFP attacks from the perspective of traffic analysis. Extensive experiments show that, in the early traffic scenario, GQUIC is the most vulnerable to WFP attacks among GQUIC, IQUIC, and HTTPS, while IQUIC is more vulnerable than HTTPS, but the vulnerability of the three protocols is similar in the normal full traffic scenario. Features transferring analysis shows that most features are transferable between protocols when on normal full traffic scenario. However, combining with the qualitative analysis of latent feature representation, we find that the transferring is inefficient when on early traffic, as GQUIC, IQUIC, and HTTPS show the significantly different magnitude of variation in the traffic distribution on early traffic. By upgrading the one-time WFP attacks to multiple WFP Top-a attacks, we find that the attack accuracy on GQUIC and IQUIC reach 95.4% and 95.5%, respectively, with only 40 packets and just using simple features, whereas reach only 60.7% when on HTTPS. We also demonstrate that the vulnerability of IQUIC is only slightly dependent on the network environment.



## **23. Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning**

cs.CR

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2106.09820v2)

**Authors**: Shubham Jain, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstracts**: End-to-end encryption (E2EE) by messaging platforms enable people to securely and privately communicate with one another. Its widespread adoption however raised concerns that illegal content might now be shared undetected. Following the global pushback against key escrow systems, client-side scanning based on perceptual hashing has been recently proposed by tech companies, governments and researchers to detect illegal content in E2EE communications. We here propose the first framework to evaluate the robustness of perceptual hashing-based client-side scanning to detection avoidance attacks and show current systems to not be robust. More specifically, we propose three adversarial attacks--a general black-box attack and two white-box attacks for discrete cosine transform-based algorithms--against perceptual hashing algorithms. In a large-scale evaluation, we show perceptual hashing-based client-side scanning mechanisms to be highly vulnerable to detection avoidance attacks in a black-box setting, with more than 99.9\% of images successfully attacked while preserving the content of the image. We furthermore show our attack to generate diverse perturbations, strongly suggesting that straightforward mitigation strategies would be ineffective. Finally, we show that the larger thresholds necessary to make the attack harder would probably require more than one billion images to be flagged and decrypted daily, raising strong privacy concerns. Taken together, our results shed serious doubts on the robustness of perceptual hashing-based client-side scanning mechanisms currently proposed by governments, organizations, and researchers around the world.



## **24. Property Inference Attacks Against GANs**

cs.CR

To Appear in NDSS 2022

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.07608v1)

**Authors**: Junhao Zhou, Yufei Chen, Chao Shen, Yang Zhang

**Abstracts**: While machine learning (ML) has made tremendous progress during the past decade, recent research has shown that ML models are vulnerable to various security and privacy attacks. So far, most of the attacks in this field focus on discriminative models, represented by classifiers. Meanwhile, little attention has been paid to the security and privacy risks of generative models, such as generative adversarial networks (GANs). In this paper, we propose the first set of training dataset property inference attacks against GANs. Concretely, the adversary aims to infer the macro-level training dataset property, i.e., the proportion of samples used to train a target GAN with respect to a certain attribute. A successful property inference attack can allow the adversary to gain extra knowledge of the target GAN's training dataset, thereby directly violating the intellectual property of the target model owner. Also, it can be used as a fairness auditor to check whether the target GAN is trained with a biased dataset. Besides, property inference can serve as a building block for other advanced attacks, such as membership inference. We propose a general attack pipeline that can be tailored to two attack scenarios, including the full black-box setting and partial black-box setting. For the latter, we introduce a novel optimization framework to increase the attack efficacy. Extensive experiments over four representative GAN models on five property inference tasks show that our attacks achieve strong performance. In addition, we show that our attacks can be used to enhance the performance of membership inference against GANs.



## **25. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

cs.CV

accepted at NeurIPS 2021, including the appendix

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.07492v1)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of hyperparameters and pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.



## **26. Towards Interpretability of Speech Pause in Dementia Detection using Adversarial Learning**

cs.CL

**SubmitDate**: 2021-11-14    [paper-pdf](http://arxiv.org/pdf/2111.07454v1)

**Authors**: Youxiang Zhu, Bang Tran, Xiaohui Liang, John A. Batsis, Robert M. Roth

**Abstracts**: Speech pause is an effective biomarker in dementia detection. Recent deep learning models have exploited speech pauses to achieve highly accurate dementia detection, but have not exploited the interpretability of speech pauses, i.e., what and how positions and lengths of speech pauses affect the result of dementia detection. In this paper, we will study the positions and lengths of dementia-sensitive pauses using adversarial learning approaches. Specifically, we first utilize an adversarial attack approach by adding the perturbation to the speech pauses of the testing samples, aiming to reduce the confidence levels of the detection model. Then, we apply an adversarial training approach to evaluate the impact of the perturbation in training samples on the detection model. We examine the interpretability from the perspectives of model accuracy, pause context, and pause length. We found that some pauses are more sensitive to dementia than other pauses from the model's perspective, e.g., speech pauses near to the verb "is". Increasing lengths of sensitive pauses or adding sensitive pauses leads the model inference to Alzheimer's Disease, while decreasing the lengths of sensitive pauses or deleting sensitive pauses leads to non-AD.



## **27. Generating Band-Limited Adversarial Surfaces Using Neural Networks**

cs.CV

**SubmitDate**: 2021-11-14    [paper-pdf](http://arxiv.org/pdf/2111.07424v1)

**Authors**: Roee Ben Shlomo, Yevgeniy Men, Ido Imanuel

**Abstracts**: Generating adversarial examples is the art of creating a noise that is added to an input signal of a classifying neural network, and thus changing the network's classification, while keeping the noise as tenuous as possible. While the subject is well-researched in the 2D regime, it is lagging behind in the 3D regime, i.e. attacking a classifying network that works on 3D point-clouds or meshes and, for example, classifies the pose of people's 3D scans. As of now, the vast majority of papers that describe adversarial attacks in this regime work by methods of optimization. In this technical report we suggest a neural network that generates the attacks. This network utilizes PointNet's architecture with some alterations. While the previous articles on which we based our work on have to optimize each shape separately, i.e. tailor an attack from scratch for each individual input without any learning, we attempt to create a unified model that can deduce the needed adversarial example with a single forward run.



## **28. Measuring the Contribution of Multiple Model Representations in Detecting Adversarial Instances**

cs.LG

**SubmitDate**: 2021-11-13    [paper-pdf](http://arxiv.org/pdf/2111.07035v1)

**Authors**: Daniel Steinberg, Paul Munro

**Abstracts**: Deep learning models have been used for a wide variety of tasks. They are prevalent in computer vision, natural language processing, speech recognition, and other areas. While these models have worked well under many scenarios, it has been shown that they are vulnerable to adversarial attacks. This has led to a proliferation of research into ways that such attacks could be identified and/or defended against. Our goal is to explore the contribution that can be attributed to using multiple underlying models for the purpose of adversarial instance detection. Our paper describes two approaches that incorporate representations from multiple models for detecting adversarial examples. We devise controlled experiments for measuring the detection impact of incrementally utilizing additional models. For many of the scenarios we consider, the results show that performance increases with the number of underlying models used for extracting representations.



## **29. Adversarially Robust Learning for Security-Constrained Optimal Power Flow**

math.OC

Accepted at Neural Information Processing Systems (NeurIPS) 2021

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06961v1)

**Authors**: Priya L. Donti, Aayushya Agarwal, Neeraj Vijay Bedmutha, Larry Pileggi, J. Zico Kolter

**Abstracts**: In recent years, the ML community has seen surges of interest in both adversarially robust learning and implicit layers, but connections between these two areas have seldom been explored. In this work, we combine innovations from these areas to tackle the problem of N-k security-constrained optimal power flow (SCOPF). N-k SCOPF is a core problem for the operation of electrical grids, and aims to schedule power generation in a manner that is robust to potentially k simultaneous equipment outages. Inspired by methods in adversarially robust training, we frame N-k SCOPF as a minimax optimization problem - viewing power generation settings as adjustable parameters and equipment outages as (adversarial) attacks - and solve this problem via gradient-based techniques. The loss function of this minimax problem involves resolving implicit equations representing grid physics and operational decisions, which we differentiate through via the implicit function theorem. We demonstrate the efficacy of our framework in solving N-3 SCOPF, which has traditionally been considered as prohibitively expensive to solve given that the problem size depends combinatorially on the number of potential outages.



## **30. Resilient Consensus-based Multi-agent Reinforcement Learning**

cs.LG

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06776v1)

**Authors**: Martin Figura, Yixuan Lin, Ji Liu, Vijay Gupta

**Abstracts**: Adversarial attacks during training can strongly influence the performance of multi-agent reinforcement learning algorithms. It is, thus, highly desirable to augment existing algorithms such that the impact of adversarial attacks on cooperative networks is eliminated, or at least bounded. In this work, we consider a fully decentralized network, where each agent receives a local reward and observes the global state and action. We propose a resilient consensus-based actor-critic algorithm, whereby each agent estimates the team-average reward and value function, and communicates the associated parameter vectors to its immediate neighbors. We show that in the presence of Byzantine agents, whose estimation and communication strategies are completely arbitrary, the estimates of the cooperative agents converge to a bounded consensus value with probability one, provided that there are at most $H$ Byzantine agents in the neighborhood of each cooperative agent and the network is $(2H+1)$-robust. Furthermore, we prove that the policy of the cooperative agents converges with probability one to a bounded neighborhood around a local maximizer of their team-average objective function under the assumption that the policies of the adversarial agents asymptotically become stationary.



## **31. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

cs.LG

22 pages, 15 figures, 5 tables

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06628v1)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.



## **32. Characterizing and Improving the Robustness of Self-Supervised Learning through Background Augmentations**

cs.CV

Technical Report; Additional Results

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2103.12719v2)

**Authors**: Chaitanya K. Ryali, David J. Schwab, Ari S. Morcos

**Abstracts**: Recent progress in self-supervised learning has demonstrated promising results in multiple visual tasks. An important ingredient in high-performing self-supervised methods is the use of data augmentation by training models to place different augmented views of the same image nearby in embedding space. However, commonly used augmentation pipelines treat images holistically, ignoring the semantic relevance of parts of an image-e.g. a subject vs. a background-which can lead to the learning of spurious correlations. Our work addresses this problem by investigating a class of simple, yet highly effective "background augmentations", which encourage models to focus on semantically-relevant content by discouraging them from focusing on image backgrounds. Through a systematic investigation, we show that background augmentations lead to substantial improvements in performance across a spectrum of state-of-the-art self-supervised methods (MoCo-v2, BYOL, SwAV) on a variety of tasks, e.g. $\sim$+1-2% gains on ImageNet, enabling performance on par with the supervised baseline. Further, we find the improvement in limited-labels settings is even larger (up to 4.2%). Background augmentations also improve robustness to a number of distribution shifts, including natural adversarial examples, ImageNet-9, adversarial attacks, ImageNet-Renditions. We also make progress in completely unsupervised saliency detection, in the process of generating saliency masks used for background augmentations.



## **33. Distributionally Robust Trajectory Optimization Under Uncertain Dynamics via Relative Entropy Trust-Regions**

eess.SY

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2103.15388v3)

**Authors**: Hany Abdulsamad, Tim Dorau, Boris Belousov, Jia-Jie Zhu, Jan Peters

**Abstracts**: Trajectory optimization and model predictive control are essential techniques underpinning advanced robotic applications, ranging from autonomous driving to full-body humanoid control. State-of-the-art algorithms have focused on data-driven approaches that infer the system dynamics online and incorporate posterior uncertainty during planning and control. Despite their success, such approaches are still susceptible to catastrophic errors that may arise due to statistical learning biases, unmodeled disturbances, or even directed adversarial attacks. In this paper, we tackle the problem of dynamics mismatch and propose a distributionally robust optimal control formulation that alternates between two relative entropy trust-region optimization problems. Our method finds the worst-case maximum entropy Gaussian posterior over the dynamics parameters and the corresponding robust policy. Furthermore, we show that our approach admits a closed-form backward-pass for a certain class of systems. Finally, we demonstrate the resulting robustness on linear and nonlinear numerical examples.



## **34. Poisoning Knowledge Graph Embeddings via Relation Inference Patterns**

cs.LG

Joint Conference of the 59th Annual Meeting of the Association for  Computational Linguistics and the 11th International Joint Conference on  Natural Language Processing (ACL-IJCNLP 2021)

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2111.06345v1)

**Authors**: Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

**Abstracts**: We study the problem of generating data poisoning attacks against Knowledge Graph Embedding (KGE) models for the task of link prediction in knowledge graphs. To poison KGE models, we propose to exploit their inductive abilities which are captured through the relationship patterns like symmetry, inversion and composition in the knowledge graph. Specifically, to degrade the model's prediction confidence on target facts, we propose to improve the model's prediction confidence on a set of decoy facts. Thus, we craft adversarial additions that can improve the model's prediction confidence on decoy facts through different inference patterns. Our experiments demonstrate that the proposed poisoning attacks outperform state-of-art baselines on four KGE models for two publicly available datasets. We also find that the symmetry pattern based attacks generalize across all model-dataset combinations which indicates the sensitivity of KGE models to this pattern.



## **35. Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes**

cs.LG

Accepted to NeurIPS 2021 [Poster]

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2110.13541v2)

**Authors**: Sanghyun Hong, Michael-Andrei Panaitescu-Liess, Yiğitcan Kaya, Tudor Dumitraş

**Abstracts**: Quantization is a popular technique that $transforms$ the parameter representation of a neural network from floating-point numbers into lower-precision ones ($e.g.$, 8-bit integers). It reduces the memory footprint and the computational cost at inference, facilitating the deployment of resource-hungry models. However, the parameter perturbations caused by this transformation result in $behavioral$ $disparities$ between the model before and after quantization. For example, a quantized model can misclassify some test-time samples that are otherwise classified correctly. It is not known whether such differences lead to a new security vulnerability. We hypothesize that an adversary may control this disparity to introduce specific behaviors that activate upon quantization. To study this hypothesis, we weaponize quantization-aware training and propose a new training framework to implement adversarial quantization outcomes. Following this framework, we present three attacks we carry out with quantization: (i) an indiscriminate attack for significant accuracy loss; (ii) a targeted attack against specific samples; and (iii) a backdoor attack for controlling the model with an input trigger. We further show that a single compromised model defeats multiple quantization schemes, including robust quantization techniques. Moreover, in a federated learning scenario, we demonstrate that a set of malicious participants who conspire can inject our quantization-activated backdoor. Lastly, we discuss potential counter-measures and show that only re-training consistently removes the attack artifacts. Our code is available at https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation



## **36. Robust Deep Reinforcement Learning through Adversarial Loss**

cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2008.01976v2)

**Authors**: Tuomas Oikarinen, Wang Zhang, Alexandre Megretski, Luca Daniel, Tsui-Wei Weng

**Abstracts**: Recent studies have shown that deep reinforcement learning agents are vulnerable to small adversarial perturbations on the agent's inputs, which raises concerns about deploying such agents in the real world. To address this issue, we propose RADIAL-RL, a principled framework to train reinforcement learning agents with improved robustness against $l_p$-norm bounded adversarial attacks. Our framework is compatible with popular deep reinforcement learning algorithms and we demonstrate its performance with deep Q-learning, A3C and PPO. We experiment on three deep RL benchmarks (Atari, MuJoCo and ProcGen) to show the effectiveness of our robust training algorithm. Our RADIAL-RL agents consistently outperform prior methods when tested against attacks of varying strength and are more computationally efficient to train. In addition, we propose a new evaluation method called Greedy Worst-Case Reward (GWC) to measure attack agnostic robustness of deep RL agents. We show that GWC can be evaluated efficiently and is a good estimate of the reward under the worst possible sequence of adversarial attacks. All code used for our experiments is available at https://github.com/tuomaso/radial_rl_v2.



## **37. Trustworthy Medical Segmentation with Uncertainty Estimation**

eess.IV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05978v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Rasool Ghulam, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.



## **38. Robust Learning via Ensemble Density Propagation in Deep Neural Networks**

cs.LG

submitted to 2020 IEEE International Workshop on Machine Learning for  Signal Processing

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05953v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Ghulam Rasool, Nidhal C. Bouaynaya, Lyudmila Mihaylova

**Abstracts**: Learning in uncertain, noisy, or adversarial environments is a challenging task for deep neural networks (DNNs). We propose a new theoretically grounded and efficient approach for robust learning that builds upon Bayesian estimation and Variational Inference. We formulate the problem of density propagation through layers of a DNN and solve it using an Ensemble Density Propagation (EnDP) scheme. The EnDP approach allows us to propagate moments of the variational probability distribution across the layers of a Bayesian DNN, enabling the estimation of the mean and covariance of the predictive distribution at the output of the model. Our experiments using MNIST and CIFAR-10 datasets show a significant improvement in the robustness of the trained models to random noise and adversarial attacks.



## **39. Audio Attacks and Defenses against AED Systems -- A Practical Study**

cs.SD

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2106.07428v4)

**Authors**: Rodrigo dos Santos, Shirin Nilizadeh

**Abstracts**: In this paper, we evaluate deep learning-enabled AED systems against evasion attacks based on adversarial examples. We test the robustness of multiple security critical AED tasks, implemented as CNNs classifiers, as well as existing third-party Nest devices, manufactured by Google, which run their own black-box deep learning models. Our adversarial examples use audio perturbations made of white and background noises. Such disturbances are easy to create, to perform and to reproduce, and can be accessible to a large number of potential attackers, even non-technically savvy ones.   We show that an adversary can focus on audio adversarial inputs to cause AED systems to misclassify, achieving high success rates, even when we use small levels of a given type of noisy disturbance. For instance, on the case of the gunshot sound class, we achieve nearly 100% success rate when employing as little as 0.05 white noise level. Similarly to what has been previously done by works focusing on adversarial examples from the image domain as well as on the speech recognition domain. We then, seek to improve classifiers' robustness through countermeasures. We employ adversarial training and audio denoising. We show that these countermeasures, when applied to audio input, can be successful, either in isolation or in combination, generating relevant increases of nearly fifty percent in the performance of the classifiers when these are under attack.



## **40. A black-box adversarial attack for poisoning clustering**

cs.LG

18 pages, Pattern Recognition 2022

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2009.05474v4)

**Authors**: Antonio Emanuele Cinà, Alessandro Torcinovich, Marcello Pelillo

**Abstracts**: Clustering algorithms play a fundamental role as tools in decision-making and sensible automation processes. Due to the widespread use of these applications, a robustness analysis of this family of algorithms against adversarial noise has become imperative. To the best of our knowledge, however, only a few works have currently addressed this problem. In an attempt to fill this gap, in this work, we propose a black-box adversarial attack for crafting adversarial samples to test the robustness of clustering algorithms. We formulate the problem as a constrained minimization program, general in its structure and customizable by the attacker according to her capability constraints. We do not assume any information about the internal structure of the victim clustering algorithm, and we allow the attacker to query it as a service only. In the absence of any derivative information, we perform the optimization with a custom approach inspired by the Abstract Genetic Algorithm (AGA). In the experimental part, we demonstrate the sensibility of different single and ensemble clustering algorithms against our crafted adversarial samples on different scenarios. Furthermore, we perform a comparison of our algorithm with a state-of-the-art approach showing that we are able to reach or even outperform its performance. Finally, to highlight the general nature of the generated noise, we show that our attacks are transferable even against supervised algorithms such as SVMs, random forests, and neural networks.



## **41. Universal Multi-Party Poisoning Attacks**

cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/1809.03474v3)

**Authors**: Saeed Mahloujifar, Mohammad Mahmoody, Ameer Mohammed

**Abstracts**: In this work, we demonstrate universal multi-party poisoning attacks that adapt and apply to any multi-party learning process with arbitrary interaction pattern between the parties. More generally, we introduce and study $(k,p)$-poisoning attacks in which an adversary controls $k\in[m]$ of the parties, and for each corrupted party $P_i$, the adversary submits some poisoned data $\mathcal{T}'_i$ on behalf of $P_i$ that is still ``$(1-p)$-close'' to the correct data $\mathcal{T}_i$ (e.g., $1-p$ fraction of $\mathcal{T}'_i$ is still honestly generated). We prove that for any ``bad'' property $B$ of the final trained hypothesis $h$ (e.g., $h$ failing on a particular test example or having ``large'' risk) that has an arbitrarily small constant probability of happening without the attack, there always is a $(k,p)$-poisoning attack that increases the probability of $B$ from $\mu$ to by $\mu^{1-p \cdot k/m} = \mu + \Omega(p \cdot k/m)$. Our attack only uses clean labels, and it is online.   More generally, we prove that for any bounded function $f(x_1,\dots,x_n) \in [0,1]$ defined over an $n$-step random process $\mathbf{X} = (x_1,\dots,x_n)$, an adversary who can override each of the $n$ blocks with even dependent probability $p$ can increase the expected output by at least $\Omega(p \cdot \mathrm{Var}[f(\mathbf{x})])$.



## **42. Sparse Adversarial Video Attacks with Spatial Transformations**

cs.CV

The short version of this work will appear in the BMVC 2021  conference

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05468v1)

**Authors**: Ronghui Mu, Wenjie Ruan, Leandro Soriano Marcolino, Qiang Ni

**Abstracts**: In recent years, a significant amount of research efforts concentrated on adversarial attacks on images, while adversarial video attacks have seldom been explored. We propose an adversarial attack strategy on videos, called DeepSAVA. Our model includes both additive perturbation and spatial transformation by a unified optimisation framework, where the structural similarity index (SSIM) measure is adopted to measure the adversarial distance. We design an effective and novel optimisation scheme which alternatively utilizes Bayesian optimisation to identify the most influential frame in a video and Stochastic gradient descent (SGD) based optimisation to produce both additive and spatial-transformed perturbations. Doing so enables DeepSAVA to perform a very sparse attack on videos for maintaining human imperceptibility while still achieving state-of-the-art performance in terms of both attack success rate and adversarial transferability. Our intensive experiments on various types of deep neural networks and video datasets confirm the superiority of DeepSAVA.



## **43. Are Transformers More Robust Than CNNs?**

cs.CV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05464v1)

**Authors**: Yutong Bai, Jieru Mei, Alan Yuille, Cihang Xie

**Abstracts**: Transformer emerges as a powerful tool for visual recognition. In addition to demonstrating competitive performance on a broad range of visual benchmarks, recent works also argue that Transformers are much more robust than Convolutions Neural Networks (CNNs). Nonetheless, surprisingly, we find these conclusions are drawn from unfair experimental settings, where Transformers and CNNs are compared at different scales and are applied with distinct training frameworks. In this paper, we aim to provide the first fair & in-depth comparisons between Transformers and CNNs, focusing on robustness evaluations.   With our unified training setup, we first challenge the previous belief that Transformers outshine CNNs when measuring adversarial robustness. More surprisingly, we find CNNs can easily be as robust as Transformers on defending against adversarial attacks, if they properly adopt Transformers' training recipes. While regarding generalization on out-of-distribution samples, we show pre-training on (external) large-scale datasets is not a fundamental request for enabling Transformers to achieve better performance than CNNs. Moreover, our ablations suggest such stronger generalization is largely benefited by the Transformer's self-attention-like architectures per se, rather than by other training setups. We hope this work can help the community better understand and benchmark the robustness of Transformers and CNNs. The code and models are publicly available at https://github.com/ytongbai/ViTs-vs-CNNs.



## **44. Statistical Perspectives on Reliability of Artificial Intelligence Systems**

cs.SE

40 pages

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05391v1)

**Authors**: Yili Hong, Jiayi Lian, Li Xu, Jie Min, Yueyao Wang, Laura J. Freeman, Xinwei Deng

**Abstracts**: Artificial intelligence (AI) systems have become increasingly popular in many areas. Nevertheless, AI technologies are still in their developing stages, and many issues need to be addressed. Among those, the reliability of AI systems needs to be demonstrated so that the AI systems can be used with confidence by the general public. In this paper, we provide statistical perspectives on the reliability of AI systems. Different from other considerations, the reliability of AI systems focuses on the time dimension. That is, the system can perform its designed functionality for the intended period. We introduce a so-called SMART statistical framework for AI reliability research, which includes five components: Structure of the system, Metrics of reliability, Analysis of failure causes, Reliability assessment, and Test planning. We review traditional methods in reliability data analysis and software reliability, and discuss how those existing methods can be transformed for reliability modeling and assessment of AI systems. We also describe recent developments in modeling and analysis of AI reliability and outline statistical research challenges in this area, including out-of-distribution detection, the effect of the training set, adversarial attacks, model accuracy, and uncertainty quantification, and discuss how those topics can be related to AI reliability, with illustrative examples. Finally, we discuss data collection and test planning for AI reliability assessment and how to improve system designs for higher AI reliability. The paper closes with some concluding remarks.



## **45. TDGIA:Effective Injection Attacks on Graph Neural Networks**

cs.LG

KDD 2021 research track paper

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2106.06663v2)

**Authors**: Xu Zou, Qinkai Zheng, Yuxiao Dong, Xinyu Guan, Evgeny Kharlamov, Jialiang Lu, Jie Tang

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. However, recent studies have shown that GNNs are vulnerable to adversarial attacks. In this paper, we study a recently-introduced realistic attack scenario on graphs -- graph injection attack (GIA). In the GIA scenario, the adversary is not able to modify the existing link structure and node attributes of the input graph, instead the attack is performed by injecting adversarial nodes into it. We present an analysis on the topological vulnerability of GNNs under GIA setting, based on which we propose the Topological Defective Graph Injection Attack (TDGIA) for effective injection attacks. TDGIA first introduces the topological defective edge selection strategy to choose the original nodes for connecting with the injected ones. It then designs the smooth feature optimization objective to generate the features for the injected nodes. Extensive experiments on large-scale datasets show that TDGIA can consistently and significantly outperform various attack baselines in attacking dozens of defense GNN models. Notably, the performance drop on target GNNs resultant from TDGIA is more than double the damage brought by the best attack solution among hundreds of submissions on KDD-CUP 2020.



## **46. A Unified Game-Theoretic Interpretation of Adversarial Robustness**

cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2103.07364v2)

**Authors**: Jie Ren, Die Zhang, Yisen Wang, Lu Chen, Zhanpeng Zhou, Yiting Chen, Xu Cheng, Xin Wang, Meng Zhou, Jie Shi, Quanshi Zhang

**Abstracts**: This paper provides a unified view to explain different adversarial attacks and defense methods, i.e. the view of multi-order interactions between input variables of DNNs. Based on the multi-order interaction, we discover that adversarial attacks mainly affect high-order interactions to fool the DNN. Furthermore, we find that the robustness of adversarially trained DNNs comes from category-specific low-order interactions. Our findings provide a potential method to unify adversarial perturbations and robustness, which can explain the existing defense methods in a principle way. Besides, our findings also make a revision of previous inaccurate understanding of the shape bias of adversarially learned features.



## **47. Membership Inference Attacks Against Self-supervised Speech Models**

cs.CR

Submitted to ICASSP 2022. Source code available at  https://github.com/RayTzeng/s3m-membership-inference

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05113v1)

**Authors**: Wei-Cheng Tseng, Wei-Tsung Kao, Hung-yi Lee

**Abstracts**: Recently, adapting the idea of self-supervised learning (SSL) on continuous speech has started gaining attention. SSL models pre-trained on a huge amount of unlabeled audio can generate general-purpose representations that benefit a wide variety of speech processing tasks. Despite their ubiquitous deployment, however, the potential privacy risks of these models have not been well investigated. In this paper, we present the first privacy analysis on several SSL speech models using Membership Inference Attacks (MIA) under black-box access. The experiment results show that these pre-trained models are vulnerable to MIA and prone to membership information leakage with high adversarial advantage scores in both utterance-level and speaker-level. Furthermore, we also conduct several ablation studies to understand the factors that contribute to the success of MIA.



## **48. A Statistical Difference Reduction Method for Escaping Backdoor Detection**

cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05077v1)

**Authors**: Pengfei Xia, Hongjing Niu, Ziqiang Li, Bin Li

**Abstracts**: Recent studies show that Deep Neural Networks (DNNs) are vulnerable to backdoor attacks. An infected model behaves normally on benign inputs, whereas its prediction will be forced to an attack-specific target on adversarial data. Several detection methods have been developed to distinguish inputs to defend against such attacks. The common hypothesis that these defenses rely on is that there are large statistical differences between the latent representations of clean and adversarial inputs extracted by the infected model. However, although it is important, comprehensive research on whether the hypothesis must be true is lacking. In this paper, we focus on it and study the following relevant questions: 1) What are the properties of the statistical differences? 2) How to effectively reduce them without harming the attack intensity? 3) What impact does this reduction have on difference-based defenses? Our work is carried out on the three questions. First, by introducing the Maximum Mean Discrepancy (MMD) as the metric, we identify that the statistical differences of multi-level representations are all large, not just the highest level. Then, we propose a Statistical Difference Reduction Method (SDRM) by adding a multi-level MMD constraint to the loss function during training a backdoor model to effectively reduce the differences. Last, three typical difference-based detection methods are examined. The F1 scores of these defenses drop from 90%-100% on the regularly trained backdoor models to 60%-70% on the models trained with SDRM on all two datasets, four model architectures, and four attack methods. The results indicate that the proposed method can be used to enhance existing attacks to escape backdoor detection algorithms.



## **49. Tightening the Approximation Error of Adversarial Risk with Auto Loss Function Search**

cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05063v1)

**Authors**: Pengfei Xia, Ziqiang Li, Bin Li

**Abstracts**: Numerous studies have demonstrated that deep neural networks are easily misled by adversarial examples. Effectively evaluating the adversarial robustness of a model is important for its deployment in practical applications. Currently, a common type of evaluation is to approximate the adversarial risk of a model as a robustness indicator by constructing malicious instances and executing attacks. Unfortunately, there is an error (gap) between the approximate value and the true value. Previous studies manually design attack methods to achieve a smaller error, which is inefficient and may miss a better solution. In this paper, we establish the tightening of the approximation error as an optimization problem and try to solve it with an algorithm. More specifically, we first analyze that replacing the non-convex and discontinuous 0-1 loss with a surrogate loss, a necessary compromise in calculating the approximation, is one of the main reasons for the error. Then we propose AutoLoss-AR, the first method for searching loss functions for tightening the approximation error of adversarial risk. Extensive experiments are conducted in multiple settings. The results demonstrate the effectiveness of the proposed method: the best-discovered loss functions outperform the handcrafted baseline by 0.9%-2.9% and 0.7%-2.0% on MNIST and CIFAR-10, respectively. Besides, we also verify that the searched losses can be transferred to other settings and explore why they are better than the baseline by visualizing the local loss landscape.



## **50. GraphAttacker: A General Multi-Task GraphAttack Framework**

cs.LG

17 pages,9 figeures

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2101.06855v2)

**Authors**: Jinyin Chen, Dunjie Zhang, Zhaoyan Ming, Kejie Huang, Wenrong Jiang, Chen Cui

**Abstracts**: Graph neural networks (GNNs) have been successfully exploited in graph analysis tasks in many real-world applications. The competition between attack and defense methods also enhances the robustness of GNNs. In this competition, the development of adversarial training methods put forward higher requirement for the diversity of attack examples. By contrast, most attack methods with specific attack strategies are difficult to satisfy such a requirement. To address this problem, we propose GraphAttacker, a novel generic graph attack framework that can flexibly adjust the structures and the attack strategies according to the graph analysis tasks. GraphAttacker generates adversarial examples through alternate training on three key components: the multi-strategy attack generator (MAG), the similarity discriminator (SD), and the attack discriminator (AD), based on the generative adversarial network (GAN). Furthermore, we introduce a novel similarity modification rate SMR to conduct a stealthier attack considering the change of node similarity distribution. Experiments on various benchmark datasets demonstrate that GraphAttacker can achieve state-of-the-art attack performance on graph analysis tasks of node classification, graph classification, and link prediction, no matter the adversarial training is conducted or not. Moreover, we also analyze the unique characteristics of each task and their specific response in the unified attack framework. The project code is available at https://github.com/honoluluuuu/GraphAttacker.



