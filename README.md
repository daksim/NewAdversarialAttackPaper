# Latest Adversarial Attack Papers
**update at 2023-02-07 09:29:24**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. MorDIFF: Recognition Vulnerability and Attack Detectability of Face Morphing Attacks Created by Diffusion Autoencoders**

cs.CV

**SubmitDate**: 2023-02-03    [abs](http://arxiv.org/abs/2302.01843v1) [paper-pdf](http://arxiv.org/pdf/2302.01843v1)

**Authors**: Naser Damer, Meiling Fang, Patrick Siebke, Jan Niklas Kolf, Marco Huber, Fadi Boutros

**Abstract**: Investigating new methods of creating face morphing attacks is essential to foresee novel attacks and help mitigate them. Creating morphing attacks is commonly either performed on the image-level or on the representation-level. The representation-level morphing has been performed so far based on generative adversarial networks (GAN) where the encoded images are interpolated in the latent space to produce a morphed image based on the interpolated vector. Such a process was constrained by the limited reconstruction fidelity of GAN architectures. Recent advances in the diffusion autoencoder models have overcome the GAN limitations, leading to high reconstruction fidelity. This theoretically makes them a perfect candidate to perform representation-level face morphing. This work investigates using diffusion autoencoders to create face morphing attacks by comparing them to a wide range of image-level and representation-level morphs. Our vulnerability analyses on four state-of-the-art face recognition models have shown that such models are highly vulnerable to the created attacks, the MorDIFF, especially when compared to existing representation-level morphs. Detailed detectability analyses are also performed on the MorDIFF, showing that they are as challenging to detect as other morphing attacks created on the image- or representation-level. Data and morphing script are made public.



## **2. On the Optimal Spoofing Attack and Countermeasure in Satellite Navigation Systems**

eess.SP

Submitted to TIFS

**SubmitDate**: 2023-02-03    [abs](http://arxiv.org/abs/2302.01841v1) [paper-pdf](http://arxiv.org/pdf/2302.01841v1)

**Authors**: Laura Crosara, Francesco Ardizzon, Stefano Tomasin, Nicola Laurenti

**Abstract**: The threat of signal spoofing attacks against GNSS has grown in recent years and has motivated the study of anti-spoofing techniques. However, defense methods have been designed only against specific attacks. This paper introduces a general model of the spoofing attack framework in GNSS, from which optimal attack and defense strategies are derived. We consider a scenario with a legitimate receiver (Bob) testing if the received signals come from multiple legitimate space vehicles (Alice) or from an attack device (Eve). We first derive the optimal attack strategy against a Gaussian transmission from Alice, by minimizing an outer bound on the achievable error probability region of the spoofing detection test. Then, framing the spoofing and its detection as an adversarial game, we show that the Gaussian transmission and the corresponding optimal attack constitute a Nash equilibrium. Lastly, we consider the case of practical modulation schemes for Alice and derive the generalized likelihood ratio test. Numerical results validate the analytical derivations and show that the bound on the achievable error region is representative of the actual performance.



## **3. Searching for the Essence of Adversarial Perturbations**

cs.LG

**SubmitDate**: 2023-02-03    [abs](http://arxiv.org/abs/2205.15357v3) [paper-pdf](http://arxiv.org/pdf/2205.15357v3)

**Authors**: Dennis Y. Menn, Tzu-hsun Feng, Hung-yi Lee

**Abstract**: Neural networks have demonstrated state-of-the-art performance in various machine learning fields. However, the introduction of malicious perturbations in input data, known as adversarial examples, has been shown to deceive neural network predictions. This poses potential risks for real-world applications such as autonomous driving and text identification. In order to mitigate these risks, a comprehensive understanding of the mechanisms underlying adversarial examples is essential. In this study, we demonstrate that adversarial perturbations contain human-recognizable information, which is the key conspirator responsible for a neural network's incorrect prediction, in contrast to the widely held belief that human-unidentifiable characteristics play a critical role in fooling a network. This concept of human-recognizable characteristics enables us to explain key features of adversarial perturbations, including their existence, transferability among different neural networks, and increased interpretability for adversarial training. We also uncover two unique properties of adversarial perturbations that deceive neural networks: masking and generation. Additionally, a special class, the complementary class, is identified when neural networks classify input images. The presence of human-recognizable information in adversarial perturbations allows researchers to gain insight into the working principles of neural networks and may lead to the development of techniques for detecting and defending against adversarial attacks.



## **4. Deep Reinforcement Learning for Cyber System Defense under Dynamic Adversarial Uncertainties**

cs.LG

**SubmitDate**: 2023-02-03    [abs](http://arxiv.org/abs/2302.01595v1) [paper-pdf](http://arxiv.org/pdf/2302.01595v1)

**Authors**: Ashutosh Dutta, Samrat Chatterjee, Arnab Bhattacharya, Mahantesh Halappanavar

**Abstract**: Development of autonomous cyber system defense strategies and action recommendations in the real-world is challenging, and includes characterizing system state uncertainties and attack-defense dynamics. We propose a data-driven deep reinforcement learning (DRL) framework to learn proactive, context-aware, defense countermeasures that dynamically adapt to evolving adversarial behaviors while minimizing loss of cyber system operations. A dynamic defense optimization problem is formulated with multiple protective postures against different types of adversaries with varying levels of skill and persistence. A custom simulation environment was developed and experiments were devised to systematically evaluate the performance of four model-free DRL algorithms against realistic, multi-stage attack sequences. Our results suggest the efficacy of DRL algorithms for proactive cyber defense under multi-stage attack profiles and system uncertainties.



## **5. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

cs.CR

**SubmitDate**: 2023-02-03    [abs](http://arxiv.org/abs/2205.10159v3) [paper-pdf](http://arxiv.org/pdf/2205.10159v3)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstract**: Adversarial examples pose a security risk as they can alter decisions of a machine learning classifier through slight input perturbations. Certified robustness has been proposed as a mitigation where given an input $x$, a classifier returns a prediction and a radius with a provable guarantee that any perturbation to $x$ within this radius (e.g., under the $L_2$ norm) will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples within the certified radius. We show that the attack can be carried out against several linear classifiers that have exact certifiable guarantees and against neural networks with ReLU activations that have conservative certifiable guarantees. Our experiments demonstrate attack success rates over 50% on random linear classifiers, up to 23.24% on the MNIST dataset for linear SVM, and up to 15.83% on the MNIST dataset for a neural network whose certified radius was given by a verifier based on mixed integer programming. Finally, as a mitigation, we advocate the use of rounded interval arithmetic to account for rounding errors.



## **6. Defensive ML: Defending Architectural Side-channels with Adversarial Obfuscation**

cs.CR

Submitted to ICML 2023

**SubmitDate**: 2023-02-03    [abs](http://arxiv.org/abs/2302.01474v1) [paper-pdf](http://arxiv.org/pdf/2302.01474v1)

**Authors**: Hyoungwook Nam, Raghavendra Pradyumna Pothukuchi, Bo Li, Nam Sung Kim, Josep Torrellas

**Abstract**: Side-channel attacks that use machine learning (ML) for signal analysis have become prominent threats to computer security, as ML models easily find patterns in signals. To address this problem, this paper explores using Adversarial Machine Learning (AML) methods as a defense at the computer architecture layer to obfuscate side channels. We call this approach Defensive ML, and the generator to obfuscate signals, defender. Defensive ML is a workflow to design, implement, train, and deploy defenders for different environments. First, we design a defender architecture given the physical characteristics and hardware constraints of the side-channel. Next, we use our DefenderGAN structure to train the defender. Finally, we apply defensive ML to thwart two side-channel attacks: one based on memory contention and the other on application power. The former uses a hardware defender with ns-level response time that attains a high level of security with half the performance impact of a traditional scheme; the latter uses a software defender with ms-level response time that provides better security than a traditional scheme with only 70% of its power overhead.



## **7. A sliced-Wasserstein distance-based approach for out-of-class-distribution detection**

cs.CV

**SubmitDate**: 2023-02-02    [abs](http://arxiv.org/abs/2302.01459v1) [paper-pdf](http://arxiv.org/pdf/2302.01459v1)

**Authors**: Mohammad Shifat E Rabbi, Abu Hasnat Mohammad Rubaiyat, Yan Zhuang, Gustavo K Rohde

**Abstract**: There exist growing interests in intelligent systems for numerous medical imaging, image processing, and computer vision applications, such as face recognition, medical diagnosis, character recognition, and self-driving cars, among others. These applications usually require solving complex classification problems involving complex images with unknown data generative processes. In addition to recent successes of the current classification approaches relying on feature engineering and deep learning, several shortcomings of them, such as the lack of robustness, generalizability, and interpretability, have also been observed. These methods often require extensive training data, are computationally expensive, and are vulnerable to out-of-distribution samples, e.g., adversarial attacks. Recently, an accurate, data-efficient, computationally efficient, and robust transport-based classification approach has been proposed, which describes a generative model-based problem formulation and closed-form solution for a specific category of classification problems. However, all these approaches lack mechanisms to detect test samples outside the class distributions used during training. In real-world settings, where the collected training samples are unable to exhaust or cover all classes, the traditional classification schemes are unable to handle the unseen classes effectively, which is especially an important issue for safety-critical systems, such as self-driving and medical imaging diagnosis. In this work, we propose a method for detecting out-of-class distributions based on the distribution of sliced-Wasserstein distance from the Radon Cumulative Distribution Transform (R-CDT) subspace. We tested our method on the MNIST and two medical image datasets and reported better accuracy than the state-of-the-art methods without an out-of-class distribution detection procedure.



## **8. Universal Secure Source Encryption under Side-Channel Attacks**

cs.IT

11 pages, 3 figures. arXiv admin note: substantial text overlap with  arXiv:2201.11670, arXiv:1801.02563

**SubmitDate**: 2023-02-02    [abs](http://arxiv.org/abs/2302.01314v1) [paper-pdf](http://arxiv.org/pdf/2302.01314v1)

**Authors**: Yasutada Oohama, Bagus Santoso

**Abstract**: We study the universal coding under side-channel attacks posed and investigated by Santoso and Oohama (2021). They proposed a theoretical security model for Shannon cipher system under side-channel attacks, where the adversary is not only allowed to collect ciphertexts by eavesdropping the public communication channel, but is also allowed to collect the physical information leaked by the devices where the cipher system is implemented on such as running time, power consumption, electromagnetic radiation, etc. For any distributions of the plain text, any noisy channels through which the adversary observe the corrupted version of the key, and any measurement device used for collecting the physical information, we can derive an achievable rate region for reliability and security such that if we compress the ciphertext using an affine encoder with rate within the achievable rate region, then: (1) anyone with secret key will be able to decrypt and decode the ciphertext correctly, but (2) any adversary who obtains the ciphertext and also the side physical information will not be able to obtain any information about the hidden source as long as the leaked physical information is encoded with a rate within the rate region.



## **9. A Light Recipe to Train Robust Vision Transformers**

cs.CV

Camera-ready version for SaTML 2023, code available at  https://github.com/dedeswim/vits-robustness-torch

**SubmitDate**: 2023-02-02    [abs](http://arxiv.org/abs/2209.07399v2) [paper-pdf](http://arxiv.org/pdf/2209.07399v2)

**Authors**: Edoardo Debenedetti, Vikash Sehwag, Prateek Mittal

**Abstract**: In this paper, we ask whether Vision Transformers (ViTs) can serve as an underlying architecture for improving the adversarial robustness of machine learning models against evasion attacks. While earlier works have focused on improving Convolutional Neural Networks, we show that also ViTs are highly suitable for adversarial training to achieve competitive performance. We achieve this objective using a custom adversarial training recipe, discovered using rigorous ablation studies on a subset of the ImageNet dataset. The canonical training recipe for ViTs recommends strong data augmentation, in part to compensate for the lack of vision inductive bias of attention modules, when compared to convolutions. We show that this recipe achieves suboptimal performance when used for adversarial training. In contrast, we find that omitting all heavy data augmentation, and adding some additional bag-of-tricks ($\varepsilon$-warmup and larger weight decay), significantly boosts the performance of robust ViTs. We show that our recipe generalizes to different classes of ViT architectures and large-scale models on full ImageNet-1k. Additionally, investigating the reasons for the robustness of our models, we show that it is easier to generate strong attacks during training when using our recipe and that this leads to better robustness at test time. Finally, we further study one consequence of adversarial training by proposing a way to quantify the semantic nature of adversarial perturbations and highlight its correlation with the robustness of the model. Overall, we recommend that the community should avoid translating the canonical training recipes in ViTs to robust training and rethink common training choices in the context of adversarial training.



## **10. Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense**

cs.CV

**SubmitDate**: 2023-02-02    [abs](http://arxiv.org/abs/2302.01056v1) [paper-pdf](http://arxiv.org/pdf/2302.01056v1)

**Authors**: Zunzhi You, Daochang Liu, Chang Xu

**Abstract**: Masked Image Modeling (MIM) has been a prevailing framework for self-supervised visual representation learning. Within the pretraining-finetuning paradigm, the MIM framework trains an encoder by reconstructing masked image patches with the help of a decoder which would be abandoned when the encoder is used for finetuning. Despite its state-of-the-art performance on clean images, MIM models are vulnerable to adversarial attacks, limiting its real-world application, and few studies have focused on this issue. In this paper, we have discovered that noisy image modeling (NIM), a variant of MIM that uses denoising as the pre-text task, provides not only good pretrained visual features, but also effective adversarial defense for downstream models. To achieve a better accuracy-robustness trade-off, we further propose to sample the hyperparameter that controls the reconstruction difficulty from random distributions instead of setting it globally, and fine-tune downstream networks with denoised images. Experimental results demonstrate that our pre-trained denoising autoencoders are effective against different white-box, gray-box, and black-box attacks without being trained with adversarial images, while not harming the clean accuracy of fine-tuned models. Source code and models will be made available.



## **11. TransFool: An Adversarial Attack against Neural Machine Translation Models**

cs.CL

**SubmitDate**: 2023-02-02    [abs](http://arxiv.org/abs/2302.00944v1) [paper-pdf](http://arxiv.org/pdf/2302.00944v1)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Deep neural networks have been shown to be vulnerable to small perturbations of their inputs, known as adversarial attacks. In this paper, we investigate the vulnerability of Neural Machine Translation (NMT) models to adversarial attacks and propose a new attack algorithm called TransFool. To fool NMT models, TransFool builds on a multi-term optimization problem and a gradient projection step. By integrating the embedding representation of a language model, we generate fluent adversarial examples in the source language that maintain a high level of semantic similarity with the clean samples. Experimental results demonstrate that, for different translation tasks and NMT architectures, our white-box attack can severely degrade the translation quality while the semantic similarity between the original and the adversarial sentences stays high. Moreover, we show that TransFool is transferable to unknown target models. Finally, based on automatic and human evaluations, TransFool leads to improvement in terms of success rate, semantic similarity, and fluency compared to the existing attacks both in white-box and black-box settings. Thus, TransFool permits us to better characterize the vulnerability of NMT models and outlines the necessity to design strong defense mechanisms and more robust NMT systems for real-life applications.



## **12. Adversarially Robust Neural Architectures**

cs.CV

13 pages, 5 figures, 8 tables

**SubmitDate**: 2023-02-02    [abs](http://arxiv.org/abs/2009.00902v2) [paper-pdf](http://arxiv.org/pdf/2009.00902v2)

**Authors**: Minjing Dong, Yanxi Li, Yunhe Wang, Chang Xu

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks. Existing methods are devoted to developing various robust training strategies or regularizations to update the weights of the neural network. But beyond the weights, the overall structure and information flow in the network are explicitly determined by the neural architecture, which remains unexplored. This paper thus aims to improve the adversarial robustness of the network from the architecture perspective. We explore the relationship among adversarial robustness, Lipschitz constant, and architecture parameters and show that an appropriate constraint on architecture parameters could reduce the Lipschitz constant to further improve the robustness. The importance of architecture parameters could vary from operation to operation or connection to connection. We approximate the Lipschitz constant of the entire network through a univariate log-normal distribution, whose mean and variance are related to architecture parameters. The confidence can be fulfilled through formulating a constraint on the distribution parameters based on the cumulative function. Compared with adversarially trained neural architectures searched by various NAS algorithms as well as efficient human-designed models, our algorithm empirically achieves the best performance among all the models under various attacks on different datasets.



## **13. Adapting Step-size: A Unified Perspective to Analyze and Improve Gradient-based Methods for Adversarial Attacks**

cs.LG

**SubmitDate**: 2023-02-02    [abs](http://arxiv.org/abs/2301.11546v3) [paper-pdf](http://arxiv.org/pdf/2301.11546v3)

**Authors**: Wei Tao, Lei Bao, Sheng Long, Gaowei Wu, Qing Tao

**Abstract**: Learning adversarial examples can be formulated as an optimization problem of maximizing the loss function with some box-constraints. However, for solving this induced optimization problem, the state-of-the-art gradient-based methods such as FGSM, I-FGSM and MI-FGSM look different from their original methods especially in updating the direction, which makes it difficult to understand them and then leaves some theoretical issues to be addressed in viewpoint of optimization. In this paper, from the perspective of adapting step-size, we provide a unified theoretical interpretation of these gradient-based adversarial learning methods. We show that each of these algorithms is in fact a specific reformulation of their original gradient methods but using the step-size rules with only current gradient information. Motivated by such analysis, we present a broad class of adaptive gradient-based algorithms based on the regular gradient methods, in which the step-size strategy utilizing information of the accumulated gradients is integrated. Such adaptive step-size strategies directly normalize the scale of the gradients rather than use some empirical operations. The important benefit is that convergence for the iterative algorithms is guaranteed and then the whole optimization process can be stabilized. The experiments demonstrate that our AdaI-FGM consistently outperforms I-FGSM and AdaMI-FGM remains competitive with MI-FGSM for black-box attacks.



## **14. Universal Soldier: Using Universal Adversarial Perturbations for Detecting Backdoor Attacks**

cs.LG

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2302.00747v1) [paper-pdf](http://arxiv.org/pdf/2302.00747v1)

**Authors**: Xiaoyun Xu, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Deep learning models achieve excellent performance in numerous machine learning tasks. Yet, they suffer from security-related issues such as adversarial examples and poisoning (backdoor) attacks. A deep learning model may be poisoned by training with backdoored data or by modifying inner network parameters. Then, a backdoored model performs as expected when receiving a clean input, but it misclassifies when receiving a backdoored input stamped with a pre-designed pattern called "trigger". Unfortunately, it is difficult to distinguish between clean and backdoored models without prior knowledge of the trigger. This paper proposes a backdoor detection method by utilizing a special type of adversarial attack, universal adversarial perturbation (UAP), and its similarities with a backdoor trigger. We observe an intuitive phenomenon: UAPs generated from backdoored models need fewer perturbations to mislead the model than UAPs from clean models. UAPs of backdoored models tend to exploit the shortcut from all classes to the target class, built by the backdoor trigger. We propose a novel method called Universal Soldier for Backdoor detection (USB) and reverse engineering potential backdoor triggers via UAPs. Experiments on 345 models trained on several datasets show that USB effectively detects the injected backdoor and provides comparable or better results than state-of-the-art methods.



## **15. Reverse engineering adversarial attacks with fingerprints from adversarial examples**

cs.AI

8 pages, 6 figures

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2301.13869v2) [paper-pdf](http://arxiv.org/pdf/2301.13869v2)

**Authors**: David Aaron Nicholson, Vincent Emanuele

**Abstract**: In spite of intense research efforts, deep neural networks remain vulnerable to adversarial examples: an input that forces the network to confidently produce incorrect outputs. Adversarial examples are typically generated by an attack algorithm that optimizes a perturbation added to a benign input. Many such algorithms have been developed. If it were possible to reverse engineer attack algorithms from adversarial examples, this could deter bad actors because of the possibility of attribution. Here we formulate reverse engineering as a supervised learning problem where the goal is to assign an adversarial example to a class that represents the algorithm and parameters used. To our knowledge it has not been previously shown whether this is even possible. We first test whether we can classify the perturbations added to images by attacks on undefended single-label image classification models. Taking a "fight fire with fire" approach, we leverage the sensitivity of deep neural networks to adversarial examples, training them to classify these perturbations. On a 17-class dataset (5 attacks, 4 bounded with 4 epsilon values each), we achieve an accuracy of 99.4% with a ResNet50 model trained on the perturbations. We then ask whether we can perform this task without access to the perturbations, obtaining an estimate of them with signal processing algorithms, an approach we call "fingerprinting". We find the JPEG algorithm serves as a simple yet effective fingerprinter (85.05% accuracy), providing a strong baseline for future work. We discuss how our approach can be extended to attack agnostic, learnable fingerprints, and to open-world scenarios with unknown attacks.



## **16. Effectiveness of Moving Target Defenses for Adversarial Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2302.00537v1) [paper-pdf](http://arxiv.org/pdf/2302.00537v1)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Several moving target defenses (MTDs) to counter adversarial ML attacks have been proposed in recent years. MTDs claim to increase the difficulty for the attacker in conducting attacks by regularly changing certain elements of the defense, such as cycling through configurations. To examine these claims, we study for the first time the effectiveness of several recent MTDs for adversarial ML attacks applied to the malware detection domain. Under different threat models, we show that transferability and query attack strategies can achieve high levels of evasion against these defenses through existing and novel attack strategies across Android and Windows. We also show that fingerprinting and reconnaissance are possible and demonstrate how attackers may obtain critical defense hyperparameters as well as information about how predictions are produced. Based on our findings, we present key recommendations for future work on the development of effective MTDs for adversarial attacks in ML-based malware detection.



## **17. Illusory Attacks: Detectability Matters in Adversarial Attacks on Sequential Decision-Makers**

cs.AI

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2207.10170v2) [paper-pdf](http://arxiv.org/pdf/2207.10170v2)

**Authors**: Tim Franzmeyer, Stephen McAleer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstract**: Autonomous agents deployed in the real world need to be robust against adversarial attacks on sensory inputs. Robustifying agent policies requires anticipating the strongest attacks possible. We demonstrate that existing observation-space attacks on reinforcement learning agents have a common weakness: while effective, their lack of temporal consistency makes them detectable using automated means or human inspection. Detectability is undesirable to adversaries as it may trigger security escalations. We introduce perfect illusory attacks, a novel form of adversarial attack on sequential decision-makers that is both effective and provably statistically undetectable. We then propose the more versatile E-illusory attacks, which result in observation transitions that are consistent with the state-transition function of the environment and can be learned end-to-end. Compared to existing attacks, we empirically find E-illusory attacks to be significantly harder to detect with automated methods, and a small study with human subjects suggests they are similarly harder to detect for humans. We conclude that future work on adversarial robustness of \mbox{(human-)AI} systems should focus on defences against attacks that are hard to detect by design.



## **18. Exploring Semantic Perturbations on Grover**

cs.LG

15 pages, 12 figures, 1 table, capstone research in machine learning

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2302.00509v1) [paper-pdf](http://arxiv.org/pdf/2302.00509v1)

**Authors**: Pranav Kulkarni, Ziqing Ji, Yan Xu, Marko Neskovic, Kevin Nolan

**Abstract**: With news and information being as easy to access as they currently are, it is more important than ever to ensure that people are not mislead by what they read. Recently, the rise of neural fake news (AI-generated fake news) and its demonstrated effectiveness at fooling humans has prompted the development of models to detect it. One such model is the Grover model, which can both detect neural fake news to prevent it, and generate it to demonstrate how a model could be misused to fool human readers. In this work we explore the Grover model's fake news detection capabilities by performing targeted attacks through perturbations on input news articles. Through this we test Grover's resilience to these adversarial attacks and expose some potential vulnerabilities which should be addressed in further iterations to ensure it can detect all types of fake news accurately.



## **19. Image Masking for Robust Self-Supervised Monocular Depth Estimation**

cs.CV

Accepted at 2023 IEEE International Conference on Robotics and  Automation (ICRA)

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2210.02357v2) [paper-pdf](http://arxiv.org/pdf/2210.02357v2)

**Authors**: Hemang Chawla, Kishaan Jeeveswaran, Elahe Arani, Bahram Zonooz

**Abstract**: Self-supervised monocular depth estimation is a salient task for 3D scene understanding. Learned jointly with monocular ego-motion estimation, several methods have been proposed to predict accurate pixel-wise depth without using labeled data. Nevertheless, these methods focus on improving performance under ideal conditions without natural or digital corruptions. The general absence of occlusions is assumed even for object-specific depth estimation. These methods are also vulnerable to adversarial attacks, which is a pertinent concern for their reliable deployment in robots and autonomous driving systems. We propose MIMDepth, a method that adapts masked image modeling (MIM) for self-supervised monocular depth estimation. While MIM has been used to learn generalizable features during pre-training, we show how it could be adapted for direct training of monocular depth estimation. Our experiments show that MIMDepth is more robust to noise, blur, weather conditions, digital artifacts, occlusions, as well as untargeted and targeted adversarial attacks.



## **20. Disco Intelligent Reflecting Surfaces: Active Channel Aging for Fully-Passive Jamming Attacks**

eess.SP

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2302.00415v1) [paper-pdf](http://arxiv.org/pdf/2302.00415v1)

**Authors**: Huan Huang, Ying Zhang, Hongliang Zhang, Yi Cai, A. Lee Swindlehurst, Zhu Han

**Abstract**: Due to the open communications environment in wireless channels, wireless networks are vulnerable to jamming attacks. However, existing approaches for jamming rely on knowledge of the legitimate users' (LUs') channels, extra jamming power, or both. To raise concerns about the potential threats posed by illegitimate intelligent reflecting surfaces (IRSs), we propose an alternative method to launch jamming attacks on LUs without either LU channel state information (CSI) or jamming power. The proposed approach employs an adversarial IRS with random phase shifts, referred to as a "disco" IRS (DIRS), that acts like a "disco ball" to actively age the LUs' channels. Such active channel aging (ACA) interference can be used to launch jamming attacks on multi-user multiple-input single-output (MU-MISO) systems. The proposed DIRS-based fully-passive jammer (FPJ) can jam LUs with no additional jamming power or knowledge of the LU CSI, and it can not be mitigated by classical anti-jamming approaches. A theoretical analysis of the proposed DIRS-based FPJ that provides an evaluation of the DIRS-based jamming attacks is derived. Based on this detailed theoretical analysis, some unique properties of the proposed DIRS-based FPJ can be obtained. Furthermore, a design example of the proposed DIRS-based FPJ based on one-bit quantization of the IRS phases is demonstrated to be sufficient for implementing the jamming attack. In addition, numerical results are provided to show the effectiveness of the derived theoretical analysis and the jamming impact of the proposed DIRS-based FPJ.



## **21. Do Perceptually Aligned Gradients Imply Adversarial Robustness?**

cs.CV

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2207.11378v2) [paper-pdf](http://arxiv.org/pdf/2207.11378v2)

**Authors**: Roy Ganz, Bahjat Kawar, Michael Elad

**Abstract**: Adversarially robust classifiers possess a trait that non-robust models do not -- Perceptually Aligned Gradients (PAG). Their gradients with respect to the input align well with human perception. Several works have identified PAG as a byproduct of robust training, but none have considered it as a standalone phenomenon nor studied its own implications. In this work, we focus on this trait and test whether \emph{Perceptually Aligned Gradients imply Robustness}. To this end, we develop a novel objective to directly promote PAG in training classifiers and examine whether models with such gradients are more robust to adversarial attacks. Extensive experiments on multiple datasets and architectures validate that models with aligned gradients exhibit significant robustness, exposing the surprising bidirectional connection between PAG and robustness. Lastly, we show that better gradient alignment leads to increased robustness and harness this observation to boost the robustness of existing adversarial training techniques.



## **22. Adversarial Driving: Attacking End-to-End Autonomous Driving**

cs.CV

7 pages, 5 figures

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2103.09151v5) [paper-pdf](http://arxiv.org/pdf/2103.09151v5)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: As research in deep neural networks advances, deep convolutional networks become promising for autonomous driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for autonomous driving. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. While for regression tasks, the effect of adversarial attacks is not as well understood. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving models. The driving system uses a regression model that takes an image as input and outputs the steering angle. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. Both attacks can be initiated in real-time on CPUs without employing GPUs. The efficiency of the attacks is illustrated using experiments conducted in Udacity Simulator. Demo video: https://youtu.be/I0i8uN2oOP0.



## **23. Adversarial Detection: Attacking Object Detection in Real Time**

cs.AI

7 pages, 9 figures

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2209.01962v3) [paper-pdf](http://arxiv.org/pdf/2209.01962v3)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90\% within about 20 iterations. The demo video is available at https://youtu.be/zJZ1aNlXsMU.



## **24. What is the Solution for State-Adversarial Multi-Agent Reinforcement Learning?**

cs.AI

**SubmitDate**: 2023-02-01    [abs](http://arxiv.org/abs/2212.02705v3) [paper-pdf](http://arxiv.org/pdf/2212.02705v3)

**Authors**: Songyang Han, Sanbao Su, Sihong He, Shuo Han, Haizhao Yang, Fei Miao

**Abstract**: Various methods for Multi-Agent Reinforcement Learning (MARL) have been developed with the assumption that agents' policies are based on accurate state information. However, policies learned through Deep Reinforcement Learning (DRL) are susceptible to adversarial state perturbation attacks. In this work, we propose a State-Adversarial Markov Game (SAMG) and make the first attempt to investigate the fundamental properties of MARL under state uncertainties. Our analysis shows that the commonly used solution concepts of optimal agent policy and robust Nash equilibrium do not always exist in SAMGs. To circumvent this difficulty, we consider a new solution concept called robust agent policy, where agents aim to maximize the worst-case expected state value. We prove the existence of robust agent policy for finite state and finite action SAMGs. Additionally, we propose a Robust Multi-Agent Adversarial Actor-Critic (RMA3C) algorithm to learn robust policies for MARL agents under state uncertainties. Our experiments demonstrate that our algorithm outperforms existing methods when faced with state perturbations and greatly improves the robustness of MARL policies. Our code is public on https://songyanghan.github.io/what_is_solution/.



## **25. The Impacts of Unanswerable Questions on the Robustness of Machine Reading Comprehension Models**

cs.AI

Accepted atThe 17th Conference of the European Chapter of the  Association for Computational Linguistics (EACL 2023)

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2302.00094v1) [paper-pdf](http://arxiv.org/pdf/2302.00094v1)

**Authors**: Son Quoc Tran, Phong Nguyen-Thuan Do, Uyen Le, Matt Kretchmar

**Abstract**: Pretrained language models have achieved super-human performances on many Machine Reading Comprehension (MRC) benchmarks. Nevertheless, their relative inability to defend against adversarial attacks has spurred skepticism about their natural language understanding. In this paper, we ask whether training with unanswerable questions in SQuAD 2.0 can help improve the robustness of MRC models against adversarial attacks. To explore that question, we fine-tune three state-of-the-art language models on either SQuAD 1.1 or SQuAD 2.0 and then evaluate their robustness under adversarial attacks. Our experiments reveal that current models fine-tuned on SQuAD 2.0 do not initially appear to be any more robust than ones fine-tuned on SQuAD 1.1, yet they reveal a measure of hidden robustness that can be leveraged to realize actual performance gains. Furthermore, we find that the robustness of models fine-tuned on SQuAD 2.0 extends to additional out-of-domain datasets. Finally, we introduce a new adversarial attack to reveal artifacts of SQuAD 2.0 that current MRC models are learning.



## **26. EC-CFI: Control-Flow Integrity via Code Encryption Counteracting Fault Attacks**

cs.CR

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2301.13760v1) [paper-pdf](http://arxiv.org/pdf/2301.13760v1)

**Authors**: Pascal Nasahl, Salmin Sultana, Hans Liljestrand, Karanvir Grewal, Michael LeMay, David M. Durham, David Schrammel, Stefan Mangard

**Abstract**: Fault attacks enable adversaries to manipulate the control-flow of security-critical applications. By inducing targeted faults into the CPU, the software's call graph can be escaped and the control-flow can be redirected to arbitrary functions inside the program. To protect the control-flow from these attacks, dedicated fault control-flow integrity (CFI) countermeasures are commonly deployed. However, these schemes either have high detection latencies or require intrusive hardware changes.   In this paper, we present EC-CFI, a software-based cryptographically enforced CFI scheme with no detection latency utilizing hardware features of recent Intel platforms. Our EC-CFI prototype is designed to prevent an adversary from escaping the program's call graph using faults by encrypting each function with a different key before execution. At runtime, the instrumented program dynamically derives the decryption key, ensuring that the code only can be successfully decrypted when the program follows the intended call graph. To enable this level of protection on Intel commodity systems, we introduce extended page table (EPT) aliasing allowing us to achieve function-granular encryption by combing Intel's TME-MK and virtualization technology. We open-source our custom LLVM-based toolchain automatically protecting arbitrary programs with EC-CFI. Furthermore, we evaluate our EPT aliasing approach with the SPEC CPU2017 and Embench-IoT benchmarks and discuss and evaluate potential TME-MK hardware changes minimizing runtime overheads.



## **27. PINCH: An Adversarial Extraction Attack Framework for Deep Learning Models**

cs.CR

19 pages, 13 figures, 5 tables

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2209.06300v2) [paper-pdf](http://arxiv.org/pdf/2209.06300v2)

**Authors**: William Hackett, Stefan Trawicki, Zhengxin Yu, Neeraj Suri, Peter Garraghan

**Abstract**: Adversarial extraction attacks constitute an insidious threat against Deep Learning (DL) models in-which an adversary aims to steal the architecture, parameters, and hyper-parameters of a targeted DL model. Existing extraction attack literature have observed varying levels of attack success for different DL models and datasets, yet the underlying cause(s) behind their susceptibility often remain unclear, and would help facilitate creating secure DL systems. In this paper we present PINCH: an efficient and automated extraction attack framework capable of designing, deploying, and analyzing extraction attack scenarios across heterogeneous hardware platforms. Using PINCH, we perform extensive experimental evaluation of extraction attacks against 21 model architectures to explore new extraction attack scenarios and further attack staging. Our findings show (1) key extraction characteristics whereby particular model configurations exhibit strong resilience against specific attacks, (2) even partial extraction success enables further staging for other adversarial attacks, and (3) equivalent stolen models uncover differences in expressive power, yet exhibit similar captured knowledge.



## **28. Are Defenses for Graph Neural Networks Robust?**

cs.LG

34 pages, 36th Conference on Neural Information Processing Systems  (NeurIPS 2022)

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2301.13694v1) [paper-pdf](http://arxiv.org/pdf/2301.13694v1)

**Authors**: Felix Mujkanovic, Simon Geisler, Stephan Günnemann, Aleksandar Bojchevski

**Abstract**: A cursory reading of the literature suggests that we have made a lot of progress in designing effective adversarial defenses for Graph Neural Networks (GNNs). Yet, the standard methodology has a serious flaw - virtually all of the defenses are evaluated against non-adaptive attacks leading to overly optimistic robustness estimates. We perform a thorough robustness analysis of 7 of the most popular defenses spanning the entire spectrum of strategies, i.e., aimed at improving the graph, the architecture, or the training. The results are sobering - most defenses show no or only marginal improvement compared to an undefended baseline. We advocate using custom adaptive attacks as a gold standard and we outline the lessons we learned from successfully designing such attacks. Moreover, our diverse collection of perturbed graphs forms a (black-box) unit test offering a first glance at a model's robustness.



## **29. Adversarial Training of Self-supervised Monocular Depth Estimation against Physical-World Attacks**

cs.CV

Accepted to ICLR2023 (Spotlight)

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2301.13487v1) [paper-pdf](http://arxiv.org/pdf/2301.13487v1)

**Authors**: Zhiyuan Cheng, James Liang, Guanhong Tao, Dongfang Liu, Xiangyu Zhang

**Abstract**: Monocular Depth Estimation (MDE) is a critical component in applications such as autonomous driving. There are various attacks against MDE networks. These attacks, especially the physical ones, pose a great threat to the security of such systems. Traditional adversarial training method requires ground-truth labels hence cannot be directly applied to self-supervised MDE that does not have ground-truth depth. Some self-supervised model hardening techniques (e.g., contrastive learning) ignore the domain knowledge of MDE and can hardly achieve optimal performance. In this work, we propose a novel adversarial training method for self-supervised MDE models based on view synthesis without using ground-truth depth. We improve adversarial robustness against physical-world attacks using L0-norm-bounded perturbation in training. We compare our method with supervised learning based and contrastive learning based methods that are tailored for MDE. Results on two representative MDE networks show that we achieve better robustness against various adversarial attacks with nearly no benign performance degradation.



## **30. Robust Linear Regression: Gradient-descent, Early-stopping, and Beyond**

stat.ML

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2301.13486v1) [paper-pdf](http://arxiv.org/pdf/2301.13486v1)

**Authors**: Meyer Scetbon, Elvis Dohmatob

**Abstract**: In this work we study the robustness to adversarial attacks, of early-stopping strategies on gradient-descent (GD) methods for linear regression. More precisely, we show that early-stopped GD is optimally robust (up to an absolute constant) against Euclidean-norm adversarial attacks. However, we show that this strategy can be arbitrarily sub-optimal in the case of general Mahalanobis attacks. This observation is compatible with recent findings in the case of classification~\cite{Vardi2022GradientMP} that show that GD provably converges to non-robust models. To alleviate this issue, we propose to apply instead a GD scheme on a transformation of the data adapted to the attack. This data transformation amounts to apply feature-depending learning rates and we show that this modified GD is able to handle any Mahalanobis attack, as well as more general attacks under some conditions. Unfortunately, choosing such adapted transformations can be hard for general attacks. To the rescue, we design a simple and tractable estimator whose adversarial risk is optimal up to within a multiplicative constant of 1.1124 in the population regime, and works for any norm.



## **31. Can we achieve robustness from data alone?**

cs.LG

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2207.11727v2) [paper-pdf](http://arxiv.org/pdf/2207.11727v2)

**Authors**: Nikolaos Tsilivis, Jingtong Su, Julia Kempe

**Abstract**: We introduce a meta-learning algorithm for adversarially robust classification. The proposed method tries to be as model agnostic as possible and optimizes a dataset prior to its deployment in a machine learning system, aiming to effectively erase its non-robust features. Once the dataset has been created, in principle no specialized algorithm (besides standard gradient descent) is needed to train a robust model. We formulate the data optimization procedure as a bi-level optimization problem on kernel regression, with a class of kernels that describe infinitely wide neural nets (Neural Tangent Kernels). We present extensive experiments on standard computer vision benchmarks using a variety of different models, demonstrating the effectiveness of our method, while also pointing out its current shortcomings. In parallel, we revisit prior work that also focused on the problem of data optimization for robust classification \citep{Ily+19}, and show that being robust to adversarial attacks after standard (gradient descent) training on a suitable dataset is more challenging than previously thought.



## **32. Certified Robustness of Learning-based Static Malware Detectors**

cs.CR

19 pages, 6 figures, 10 tables. Includes 5 pages of appendices

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2302.01757v1) [paper-pdf](http://arxiv.org/pdf/2302.01757v1)

**Authors**: Zhuoqun Huang, Neil G. Marchant, Keane Lucas, Lujo Bauer, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstract**: Certified defenses are a recent development in adversarial machine learning (ML), which aim to rigorously guarantee the robustness of ML models to adversarial perturbations. A large body of work studies certified defenses in computer vision, where $\ell_p$ norm-bounded evasion attacks are adopted as a tractable threat model. However, this threat model has known limitations in vision, and is not applicable to other domains -- e.g., where inputs may be discrete or subject to complex constraints. Motivated by this gap, we study certified defenses for malware detection, a domain where attacks against ML-based systems are a real and current threat. We consider static malware detection systems that operate on byte-level data. Our certified defense is based on the approach of randomized smoothing which we adapt by: (1) replacing the standard Gaussian randomization scheme with a novel deletion randomization scheme that operates on bytes or chunks of an executable; and (2) deriving a certificate that measures robustness to evasion attacks in terms of generalized edit distance. To assess the size of robustness certificates that are achievable while maintaining high accuracy, we conduct experiments on malware datasets using a popular convolutional malware detection model, MalConv. We are able to accurately classify 91% of the inputs while being certifiably robust to any adversarial perturbations of edit distance 128 bytes or less. By comparison, an existing certification of up to 128 bytes of substitutions (without insertions or deletions) achieves an accuracy of 78%. In addition, given that robustness certificates are conservative, we evaluate practical robustness to several recently published evasion attacks and, in some cases, find robustness beyond certified guarantees.



## **33. Inference Time Evidences of Adversarial Attacks for Forensic on Transformers**

cs.CV

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2301.13356v1) [paper-pdf](http://arxiv.org/pdf/2301.13356v1)

**Authors**: Hugo Lemarchant, Liangzi Li, Yiming Qian, Yuta Nakashima, Hajime Nagahara

**Abstract**: Vision Transformers (ViTs) are becoming a very popular paradigm for vision tasks as they achieve state-of-the-art performance on image classification. However, although early works implied that this network structure had increased robustness against adversarial attacks, some works argue ViTs are still vulnerable. This paper presents our first attempt toward detecting adversarial attacks during inference time using the network's input and outputs as well as latent features. We design four quantifications (or derivatives) of input, output, and latent vectors of ViT-based models that provide a signature of the inference, which could be beneficial for the attack detection, and empirically study their behavior over clean samples and adversarial samples. The results demonstrate that the quantifications from input (images) and output (posterior probabilities) are promising for distinguishing clean and adversarial samples, while latent vectors offer less discriminative power, though they give some insights on how adversarial perturbations work.



## **34. Affinity Uncertainty-based Hard Negative Mining in Graph Contrastive Learning**

cs.LG

11 pages, 4 figures

**SubmitDate**: 2023-01-31    [abs](http://arxiv.org/abs/2301.13340v1) [paper-pdf](http://arxiv.org/pdf/2301.13340v1)

**Authors**: Chaoxi Niu, Guansong Pang, Ling Chen

**Abstract**: Hard negative mining has shown effective in enhancing self-supervised contrastive learning (CL) on diverse data types, including graph contrastive learning (GCL). Existing hardness-aware CL methods typically treat negative instances that are most similar to the anchor instance as hard negatives, which helps improve the CL performance, especially on image data. However, this approach often fails to identify the hard negatives but leads to many false negatives on graph data. This is mainly due to that the learned graph representations are not sufficiently discriminative due to over-smooth representations and/or non-i.i.d. issues in graph data. To tackle this problem, this paper proposes a novel approach that builds a discriminative model on collective affinity information (i.e, two sets of pairwise affinities between the negative instances and the anchor instance) to mine hard negatives in GCL. In particular, the proposed approach evaluates how confident/uncertain the discriminative model is about the affinity of each negative instance to an anchor instance to determine its hardness weight relative to the anchor instance. This uncertainty information is then incorporated into existing GCL loss functions via a weighting term to enhance their performance. The enhanced GCL is theoretically grounded that the resulting GCL loss is equivalent to a triplet loss with an adaptive margin being exponentially proportional to the learned uncertainty of each negative instance. Extensive experiments on 10 graph datasets show that our approach i) consistently enhances different state-of-the-art GCL methods in both graph and node classification tasks, and ii) significantly improves their robustness against adversarial attacks.



## **35. E-DPNCT: An Enhanced Attack Resilient Differential Privacy Model For Smart Grids Using Split Noise Cancellation**

cs.CR

13 pages, 7 figues, 1 tables

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2110.11091v3) [paper-pdf](http://arxiv.org/pdf/2110.11091v3)

**Authors**: Khadija Hafeez, Donna OShea, Thomas Newe, Mubashir Husain Rehmani

**Abstract**: High frequency reporting of energy consumption data in smart grids can be used to infer sensitive information regarding the consumers life style and poses serious security and privacy threats. Differential privacy (DP) based privacy models for smart grids ensure privacy when analysing energy consumption data for billing and load monitoring. However, DP models for smart grids are vulnerable to collusion attack where an adversary colludes with malicious smart meters and un-trusted aggregator in order to get private information from other smart meters. We propose an Enhanced Differential Private Noise Cancellation Model for Load Monitoring and Billing for Smart Meters (E-DPNCT) to protect the privacy of the smart grid data using a split noise cancellation protocol with multiple master smart meters (MSMs) to provide accurate billing and load monitoring and resistance against collusion attacks. We did extensive comparison of our E-DPNCT model with state of the art attack resistant privacy preserving models such as EPIC for collusion attack. We simulate our E-DPNCT model with real time data which shows significant improvement in privacy attack scenarios. Further, we analyze the impact of selecting different sensitivity parameters for calibrating DP noise over the privacy of customer electricity profile and accuracy of electricity data aggregation such as load monitoring and billing.



## **36. Towards Adversarial Realism and Robust Learning for IoT Intrusion Detection and Classification**

cs.CR

19 pages, 5 tables, 7 figures, Internet of Things journal

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.13122v1) [paper-pdf](http://arxiv.org/pdf/2301.13122v1)

**Authors**: João Vitorino, Isabel Praça, Eva Maia

**Abstract**: The Internet of Things (IoT) faces tremendous security challenges. Machine learning models can be used to tackle the growing number of cyber-attack variations targeting IoT systems, but the increasing threat posed by adversarial attacks restates the need for reliable defense strategies. This work describes the types of constraints required for an adversarial cyber-attack example to be realistic and proposes a methodology for a trustworthy adversarial robustness analysis with a realistic adversarial evasion attack vector. The proposed methodology was used to evaluate three supervised algorithms, Random Forest (RF), Extreme Gradient Boosting (XGB), and Light Gradient Boosting Machine (LGBM), and one unsupervised algorithm, Isolation Forest (IFOR). Constrained adversarial examples were generated with the Adaptative Perturbation Pattern Method (A2PM), and evasion attacks were performed against models created with regular and adversarial training. Even though RF was the least affected in binary classification, XGB consistently achieved the highest accuracy in multi-class classification. The obtained results evidence the inherent susceptibility of tree-based algorithms and ensembles to adversarial evasion attacks and demonstrates the benefits of adversarial training and a security by design approach for a more robust IoT network intrusion detection.



## **37. Anchor-Based Adversarially Robust Zero-Shot Learning Driven by Language**

cs.CV

11 pages

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.13096v1) [paper-pdf](http://arxiv.org/pdf/2301.13096v1)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. We consider adversarial defense in the case of zero-shot image classification setting, which has rarely been explored because both adversarial defense and zero-shot learning are challenging. We propose LAAT, a novel Language-driven, Anchor-based Adversarial Training strategy, to improve the adversarial robustness in a zero-shot setting. LAAT uses a text encoder to obtain fixed anchors (normalized feature embeddings) of each category, then uses these anchors to perform adversarial training. The text encoder has the property that semantically similar categories can be mapped to neighboring anchors in the feature space. By leveraging this property, LAAT can make the image model adversarially robust on novel categories without any extra examples. Experimental results show that our method achieves impressive zero-shot adversarial performance, even surpassing the previous state-of-the-art adversarially robust one-shot methods in most attacking settings. When models are trained with LAAT on large datasets like ImageNet-1K, they can have substantial zero-shot adversarial robustness across several downstream datasets.



## **38. On the Efficacy of Metrics to Describe Adversarial Attacks**

cs.LG

7 pages, selected for presentation at AICS

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.13028v1) [paper-pdf](http://arxiv.org/pdf/2301.13028v1)

**Authors**: Tommaso Puccetti, Tommaso Zoppi, Andrea Ceccarelli

**Abstract**: Adversarial defenses are naturally evaluated on their ability to tolerate adversarial attacks. To test defenses, diverse adversarial attacks are crafted, that are usually described in terms of their evading capability and the L0, L1, L2, and Linf norms. We question if the evading capability and L-norms are the most effective information to claim that defenses have been tested against a representative attack set. To this extent, we select image quality metrics from the state of the art and search correlations between image perturbation and detectability. We observe that computing L-norms alone is rarely the preferable solution. We observe a strong correlation between the identified metrics computed on an adversarial image and the output of a detector on such an image, to the extent that they can predict the response of a detector with approximately 0.94 accuracy. Further, we observe that metrics can classify attacks based on similar perturbations and similar detectability. This suggests a possible review of the approach to evaluate detectors, where additional metrics are included to assure that a representative attack dataset is selected.



## **39. PCV: A Point Cloud-Based Network Verifier**

cs.CV

11 pages, 12 figures

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.11806v2) [paper-pdf](http://arxiv.org/pdf/2301.11806v2)

**Authors**: Arup Kumar Sarker, Farzana Yasmin Ahmad, Matthew B. Dwyer

**Abstract**: 3D vision with real-time LiDAR-based point cloud data became a vital part of autonomous system research, especially perception and prediction modules use for object classification, segmentation, and detection. Despite their success, point cloud-based network models are vulnerable to multiple adversarial attacks, where the certain factor of changes in the validation set causes significant performance drop in well-trained networks. Most of the existing verifiers work perfectly on 2D convolution. Due to complex architecture, dimension of hyper-parameter, and 3D convolution, no verifiers can perform the basic layer-wise verification. It is difficult to conclude the robustness of a 3D vision model without performing the verification. Because there will be always corner cases and adversarial input that can compromise the model's effectiveness.   In this project, we describe a point cloud-based network verifier that successfully deals state of the art 3D classifier PointNet verifies the robustness by generating adversarial inputs. We have used extracted properties from the trained PointNet and changed certain factors for perturbation input. We calculate the impact on model accuracy versus property factor and can test PointNet network's robustness against a small collection of perturbing input states resulting from adversarial attacks like the suggested hybrid reverse signed attack. The experimental results reveal that the resilience property of PointNet is affected by our hybrid reverse signed perturbation strategy



## **40. Improving Adversarial Transferability with Scheduled Step Size and Dual Example**

cs.LG

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.12968v1) [paper-pdf](http://arxiv.org/pdf/2301.12968v1)

**Authors**: Zeliang Zhang, Peihan Liu, Xiaosen Wang, Chenliang Xu

**Abstract**: Deep neural networks are widely known to be vulnerable to adversarial examples, especially showing significantly poor performance on adversarial examples generated under the white-box setting. However, most white-box attack methods rely heavily on the target model and quickly get stuck in local optima, resulting in poor adversarial transferability. The momentum-based methods and their variants are proposed to escape the local optima for better transferability. In this work, we notice that the transferability of adversarial examples generated by the iterative fast gradient sign method (I-FGSM) exhibits a decreasing trend when increasing the number of iterations. Motivated by this finding, we argue that the information of adversarial perturbations near the benign sample, especially the direction, benefits more on the transferability. Thus, we propose a novel strategy, which uses the Scheduled step size and the Dual example (SD), to fully utilize the adversarial information near the benign sample. Our proposed strategy can be easily integrated with existing adversarial attack methods for better adversarial transferability. Empirical evaluations on the standard ImageNet dataset demonstrate that our proposed method can significantly enhance the transferability of existing adversarial attacks.



## **41. Identifying Adversarially Attackable and Robust Samples**

cs.LG

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.12896v1) [paper-pdf](http://arxiv.org/pdf/2301.12896v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: This work proposes a novel perspective on adversarial attacks by introducing the concept of sample attackability and robustness. Adversarial attacks insert small, imperceptible perturbations to the input that cause large, undesired changes to the output of deep learning models. Despite extensive research on generating adversarial attacks and building defense systems, there has been limited research on understanding adversarial attacks from an input-data perspective. We propose a deep-learning-based method for detecting the most attackable and robust samples in an unseen dataset for an unseen target model. The proposed method is based on a neural network architecture that takes as input a sample and outputs a measure of attackability or robustness. The proposed method is evaluated using a range of different models and different attack methods, and the results demonstrate its effectiveness in detecting the samples that are most likely to be affected by adversarial attacks. Understanding sample attackability can have important implications for future work in sample-selection tasks. For example in active learning, the acquisition function can be designed to select the most attackable samples, or in adversarial training, only the most attackable samples are selected for augmentation.



## **42. On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex**

cs.CL

Accepted at EACL2023 (main)

**SubmitDate**: 2023-02-06    [abs](http://arxiv.org/abs/2301.12868v2) [paper-pdf](http://arxiv.org/pdf/2301.12868v2)

**Authors**: Terry Yue Zhuo, Zhuang Li, Yujin Huang, Fatemeh Shiri, Weiqing Wang, Gholamreza Haffari, Yuan-Fang Li

**Abstract**: Semantic parsing is a technique aimed at constructing a structured representation of the meaning of a natural-language question. Recent advancements in few-shot language models trained on code have demonstrated superior performance in generating these representations compared to traditional unimodal language models, which are trained on downstream tasks. Despite these advancements, existing fine-tuned neural semantic parsers are susceptible to adversarial attacks on natural-language inputs. While it has been established that the robustness of smaller semantic parsers can be enhanced through adversarial training, this approach is not feasible for large language models in real-world scenarios, as it requires both substantial computational resources and expensive human annotation on in-domain semantic parsing data. This paper presents the first empirical study on the adversarial robustness of a large prompt-based language model of code, \codex. Our results demonstrate that the state-of-the-art (SOTA) code-language models are vulnerable to carefully crafted adversarial examples. To address this challenge, we propose methods for improving robustness without the need for significant amounts of labeled data or heavy computational resources.



## **43. The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance**

eess.AS

Published in IEEE/ACM Transactions on Audio, Speech, and Language  Processing (DOI: 10.1109/TASLP.2022.3233236)

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2204.05177v3) [paper-pdf](http://arxiv.org/pdf/2204.05177v3)

**Authors**: Lin Zhang, Xin Wang, Erica Cooper, Nicholas Evans, Junichi Yamagishi

**Abstract**: Automatic speaker verification is susceptible to various manipulations and spoofing, such as text-to-speech synthesis, voice conversion, replay, tampering, adversarial attacks, and so on. We consider a new spoofing scenario called "Partial Spoof" (PS) in which synthesized or transformed speech segments are embedded into a bona fide utterance. While existing countermeasures (CMs) can detect fully spoofed utterances, there is a need for their adaptation or extension to the PS scenario. We propose various improvements to construct a significantly more accurate CM that can detect and locate short-generated spoofed speech segments at finer temporal resolutions. First, we introduce newly developed self-supervised pre-trained models as enhanced feature extractors. Second, we extend our PartialSpoof database by adding segment labels for various temporal resolutions. Since the short spoofed speech segments to be embedded by attackers are of variable length, six different temporal resolutions are considered, ranging from as short as 20 ms to as large as 640 ms. Third, we propose a new CM that enables the simultaneous use of the segment-level labels at different temporal resolutions as well as utterance-level labels to execute utterance- and segment-level detection at the same time. We also show that the proposed CM is capable of detecting spoofing at the utterance level with low error rates in the PS scenario as well as in a related logical access (LA) scenario. The equal error rates of utterance-level detection on the PartialSpoof database and ASVspoof 2019 LA database were 0.77 and 0.90%, respectively.



## **44. GPS-Spoofing Attack Detection Mechanism for UAV Swarms**

cs.CR

8 pages, 3 figures

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.12766v1) [paper-pdf](http://arxiv.org/pdf/2301.12766v1)

**Authors**: Pavlo Mykytyn, Marcin Brzozowski, Zoya Dyka, Peter Langendoerfer

**Abstract**: Recently autonomous and semi-autonomous Unmanned Aerial Vehicle (UAV) swarms started to receive a lot of research interest and demand from various civil application fields. However, for successful mission execution, UAV swarms require Global navigation satellite system signals and in particular, Global Positioning System (GPS) signals for navigation. Unfortunately, civil GPS signals are unencrypted and unauthenticated, which facilitates the execution of GPS spoofing attacks. During these attacks, adversaries mimic the authentic GPS signal and broadcast it to the targeted UAV in order to change its course, and force it to land or crash. In this study, we propose a GPS spoofing detection mechanism capable of detecting single-transmitter and multi-transmitter GPS spoofing attacks to prevent the outcomes mentioned above. Our detection mechanism is based on comparing the distance between each two swarm members calculated from their GPS coordinates to the distance acquired from Impulse Radio Ultra-Wideband ranging between the same swarm members. If the difference in distances is larger than a chosen threshold the GPS spoofing attack is declared detected.



## **45. Private Node Selection in Personalized Decentralized Learning**

cs.LG

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.12755v1) [paper-pdf](http://arxiv.org/pdf/2301.12755v1)

**Authors**: Edvin Listo Zec, Johan Östman, Olof Mogren, Daniel Gillblad

**Abstract**: In this paper, we propose a novel approach for privacy-preserving node selection in personalized decentralized learning, which we refer to as Private Personalized Decentralized Learning (PPDL). Our method mitigates the risk of inference attacks through the use of secure aggregation while simultaneously enabling efficient identification of collaborators. This is achieved by leveraging adversarial multi-armed bandit optimization that exploits dependencies between the different arms. Through comprehensive experimentation on various benchmarks under label and covariate shift, we demonstrate that our privacy-preserving approach outperforms previous non-private methods in terms of model performance.



## **46. Robust Stochastic Linear Contextual Bandits Under Adversarial Attacks**

stat.ML

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2106.02978v3) [paper-pdf](http://arxiv.org/pdf/2106.02978v3)

**Authors**: Qin Ding, Cho-Jui Hsieh, James Sharpnack

**Abstract**: Stochastic linear contextual bandit algorithms have substantial applications in practice, such as recommender systems, online advertising, clinical trials, etc. Recent works show that optimal bandit algorithms are vulnerable to adversarial attacks and can fail completely in the presence of attacks. Existing robust bandit algorithms only work for the non-contextual setting under the attack of rewards and cannot improve the robustness in the general and popular contextual bandit environment. In addition, none of the existing methods can defend against attacked context. In this work, we provide the first robust bandit algorithm for stochastic linear contextual bandit setting under a fully adaptive and omniscient attack with sub-linear regret. Our algorithm not only works under the attack of rewards, but also under attacked context. Moreover, it does not need any information about the attack budget or the particular form of the attack. We provide theoretical guarantees for our proposed algorithm and show by experiments that our proposed algorithm improves the robustness against various kinds of popular attacks.



## **47. Sparse Oblique Decision Trees: A Tool to Understand and Manipulate Neural Net Features**

cs.LG

Appears in Data Mining and Knowledge Discovery (2023), Special Issue  on Explainable and Interpretable Machine Learning and Data Mining

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2104.02922v2) [paper-pdf](http://arxiv.org/pdf/2104.02922v2)

**Authors**: Suryabhan Singh Hada, Miguel Á. Carreira-Perpiñán, Arman Zharmagambetov

**Abstract**: The widespread deployment of deep nets in practical applications has lead to a growing desire to understand how and why such black-box methods perform prediction. Much work has focused on understanding what part of the input pattern (an image, say) is responsible for a particular class being predicted, and how the input may be manipulated to predict a different class. We focus instead on understanding which of the internal features computed by the neural net are responsible for a particular class. We achieve this by mimicking part of the neural net with an oblique decision tree having sparse weight vectors at the decision nodes. Using the recently proposed Tree Alternating Optimization (TAO) algorithm, we are able to learn trees that are both highly accurate and interpretable. Such trees can faithfully mimic the part of the neural net they replaced, and hence they can provide insights into the deep net black box. Further, we show we can easily manipulate the neural net features in order to make the net predict, or not predict, a given class, thus showing that it is possible to carry out adversarial attacks at the level of the features. These insights and manipulations apply globally to the entire training and test set, not just at a local (single-instance) level. We demonstrate this robustly in the MNIST and ImageNet datasets with LeNet5 and VGG networks.



## **48. Attack Impact Evaluation for Stochastic Control Systems through Alarm Flag State Augmentation**

math.OC

8 pages. arXiv admin note: substantial text overlap with  arXiv:2203.16803

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.12684v1) [paper-pdf](http://arxiv.org/pdf/2301.12684v1)

**Authors**: Hampei Sasahara, Takashi Tanaka, Henrik Sandberg

**Abstract**: This note addresses the problem of evaluating the impact of an attack on discrete-time nonlinear stochastic control systems. The problem is formulated as an optimal control problem with a joint chance constraint that forces the adversary to avoid detection throughout a given time period. Due to the joint constraint, the optimal control policy depends not only on the current state, but also on the entire history, leading to an explosion of the search space and making the problem generally intractable. However, we discover that the current state and whether an alarm has been triggered, or not, is sufficient for specifying the optimal decision at each time step. This information, which we refer to as the alarm flag, can be added to the state space to create an equivalent optimal control problem that can be solved with existing numerical approaches using a Markov policy. Additionally, we note that the formulation results in a policy that does not avoid detection once an alarm has been triggered. We extend the formulation to handle multi-alarm avoidance policies for more reasonable attack impact evaluations, and show that the idea of augmenting the state space with an alarm flag is valid in this extended formulation as well.



## **49. Feature-Space Bayesian Adversarial Learning Improved Malware Detector Robustness**

cs.CR

Accepted to AAAI 2023 conference

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.12680v1) [paper-pdf](http://arxiv.org/pdf/2301.12680v1)

**Authors**: Bao Gia Doan, Shuiqiao Yang, Paul Montague, Olivier De Vel, Tamas Abraham, Seyit Camtepe, Salil S. Kanhere, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: We present a new algorithm to train a robust malware detector. Modern malware detectors rely on machine learning algorithms. Now, the adversarial objective is to devise alterations to the malware code to decrease the chance of being detected whilst preserving the functionality and realism of the malware. Adversarial learning is effective in improving robustness but generating functional and realistic adversarial malware samples is non-trivial. Because: i) in contrast to tasks capable of using gradient-based feedback, adversarial learning in a domain without a differentiable mapping function from the problem space (malware code inputs) to the feature space is hard; and ii) it is difficult to ensure the adversarial malware is realistic and functional. This presents a challenge for developing scalable adversarial machine learning algorithms for large datasets at a production or commercial scale to realize robust malware detectors. We propose an alternative; perform adversarial learning in the feature space in contrast to the problem space. We prove the projection of perturbed, yet valid malware, in the problem space into feature space will always be a subset of adversarials generated in the feature space. Hence, by generating a robust network against feature-space adversarial examples, we inherently achieve robustness against problem-space adversarial examples. We formulate a Bayesian adversarial learning objective that captures the distribution of models for improved robustness. We prove that our learning method bounds the difference between the adversarial risk and empirical risk explaining the improved robustness. We show that adversarially trained BNNs achieve state-of-the-art robustness. Notably, adversarially trained BNNs are robust against stronger attacks with larger attack budgets by a margin of up to 15% on a recent production-scale malware dataset of more than 20 million samples.



## **50. Lateralized Learning for Multi-Class Visual Classification Tasks**

cs.CV

13 pages, 5 figures

**SubmitDate**: 2023-01-30    [abs](http://arxiv.org/abs/2301.12637v1) [paper-pdf](http://arxiv.org/pdf/2301.12637v1)

**Authors**: Abubakar Siddique, Will N. Browne, Gina M. Grimshaw

**Abstract**: The majority of computer vision algorithms fail to find higher-order (abstract) patterns in an image so are not robust against adversarial attacks, unlike human lateralized vision. Deep learning considers each input pixel in a homogeneous manner such that different parts of a ``locality-sensitive hashing table'' are often not connected, meaning higher-order patterns are not discovered. Hence these systems are not robust against noisy, irrelevant, and redundant data, resulting in the wrong prediction being made with high confidence. Conversely, vertebrate brains afford heterogeneous knowledge representation through lateralization, enabling modular learning at different levels of abstraction. This work aims to verify the effectiveness, scalability, and robustness of a lateralized approach to real-world problems that contain noisy, irrelevant, and redundant data. The experimental results of multi-class (200 classes) image classification show that the novel system effectively learns knowledge representation at multiple levels of abstraction making it more robust than other state-of-the-art techniques. Crucially, the novel lateralized system outperformed all the state-of-the-art deep learning-based systems for the classification of normal and adversarial images by 19.05% - 41.02% and 1.36% - 49.22%, respectively. Findings demonstrate the value of heterogeneous and lateralized learning for computer vision applications.



