# Latest Adversarial Attack Papers
**update at 2023-10-06 10:46:09**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. OMG-ATTACK: Self-Supervised On-Manifold Generation of Transferable Evasion Attacks**

cs.LG

ICCV 2023, AROW Workshop

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03707v1) [paper-pdf](http://arxiv.org/pdf/2310.03707v1)

**Authors**: Ofir Bar Tal, Adi Haviv, Amit H. Bermano

**Abstract**: Evasion Attacks (EA) are used to test the robustness of trained neural networks by distorting input data to misguide the model into incorrect classifications. Creating these attacks is a challenging task, especially with the ever-increasing complexity of models and datasets. In this work, we introduce a self-supervised, computationally economical method for generating adversarial examples, designed for the unseen black-box setting. Adapting techniques from representation learning, our method generates on-manifold EAs that are encouraged to resemble the data distribution. These attacks are comparable in effectiveness compared to the state-of-the-art when attacking the model trained on, but are significantly more effective when attacking unseen models, as the attacks are more related to the data rather than the model itself. Our experiments consistently demonstrate the method is effective across various models, unseen data categories, and even defended models, suggesting a significant role for on-manifold EAs when targeting unseen models.



## **2. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03684v1) [paper-pdf](http://arxiv.org/pdf/2310.03684v1)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.



## **3. Certification of Deep Learning Models for Medical Image Segmentation**

eess.IV

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03664v1) [paper-pdf](http://arxiv.org/pdf/2310.03664v1)

**Authors**: Othmane Laousy, Alexandre Araujo, Guillaume Chassagnon, Nikos Paragios, Marie-Pierre Revel, Maria Vakalopoulou

**Abstract**: In medical imaging, segmentation models have known a significant improvement in the past decade and are now used daily in clinical practice. However, similar to classification models, segmentation models are affected by adversarial attacks. In a safety-critical field like healthcare, certifying model predictions is of the utmost importance. Randomized smoothing has been introduced lately and provides a framework to certify models and obtain theoretical guarantees. In this paper, we present for the first time a certified segmentation baseline for medical imaging based on randomized smoothing and diffusion models. Our results show that leveraging the power of denoising diffusion probabilistic models helps us overcome the limits of randomized smoothing. We conduct extensive experiments on five public datasets of chest X-rays, skin lesions, and colonoscopies, and empirically show that we are able to maintain high certified Dice scores even for highly perturbed images. Our work represents the first attempt to certify medical image segmentation models, and we aspire for it to set a foundation for future benchmarks in this crucial and largely uncharted area.



## **4. Targeted Adversarial Attacks on Generalizable Neural Radiance Fields**

cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03578v1) [paper-pdf](http://arxiv.org/pdf/2310.03578v1)

**Authors**: Andras Horvath, Csaba M. Jozsa

**Abstract**: Neural Radiance Fields (NeRFs) have recently emerged as a powerful tool for 3D scene representation and rendering. These data-driven models can learn to synthesize high-quality images from sparse 2D observations, enabling realistic and interactive scene reconstructions. However, the growing usage of NeRFs in critical applications such as augmented reality, robotics, and virtual environments could be threatened by adversarial attacks.   In this paper we present how generalizable NeRFs can be attacked by both low-intensity adversarial attacks and adversarial patches, where the later could be robust enough to be used in real world applications. We also demonstrate targeted attacks, where a specific, predefined output scene is generated by these attack with success.



## **5. Enhancing Adversarial Robustness via Score-Based Optimization**

cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2307.04333v2) [paper-pdf](http://arxiv.org/pdf/2307.04333v2)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.



## **6. AdvRain: Adversarial Raindrops to Attack Camera-based Smart Vision Systems**

cs.CV

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2303.01338v2) [paper-pdf](http://arxiv.org/pdf/2303.01338v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Muhammad Shafique

**Abstract**: Vision-based perception modules are increasingly deployed in many applications, especially autonomous vehicles and intelligent robots. These modules are being used to acquire information about the surroundings and identify obstacles. Hence, accurate detection and classification are essential to reach appropriate decisions and take appropriate and safe actions at all times. Current studies have demonstrated that "printed adversarial attacks", known as physical adversarial attacks, can successfully mislead perception models such as object detectors and image classifiers. However, most of these physical attacks are based on noticeable and eye-catching patterns for generated perturbations making them identifiable/detectable by human eye or in test drives. In this paper, we propose a camera-based inconspicuous adversarial attack (\textbf{AdvRain}) capable of fooling camera-based perception systems over all objects of the same class. Unlike mask based fake-weather attacks that require access to the underlying computing hardware or image memory, our attack is based on emulating the effects of a natural weather condition (i.e., Raindrops) that can be printed on a translucent sticker, which is externally placed over the lens of a camera. To accomplish this, we provide an iterative process based on performing a random search aiming to identify critical positions to make sure that the performed transformation is adversarial for a target classifier. Our transformation is based on blurring predefined parts of the captured image corresponding to the areas covered by the raindrop. We achieve a drop in average model accuracy of more than $45\%$ and $40\%$ on VGG19 for ImageNet and Resnet34 for Caltech-101, respectively, using only $20$ raindrops.



## **7. Efficient Biologically Plausible Adversarial Training**

cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2309.17348v3) [paper-pdf](http://arxiv.org/pdf/2309.17348v3)

**Authors**: Matilde Tristany Farinha, Thomas Ortner, Giorgia Dellaferrera, Benjamin Grewe, Angeliki Pantazi

**Abstract**: Artificial Neural Networks (ANNs) trained with Backpropagation (BP) show astounding performance and are increasingly often used in performing our daily life tasks. However, ANNs are highly vulnerable to adversarial attacks, which alter inputs with small targeted perturbations that drastically disrupt the models' performance. The most effective method to make ANNs robust against these attacks is adversarial training, in which the training dataset is augmented with exemplary adversarial samples. Unfortunately, this approach has the drawback of increased training complexity since generating adversarial samples is very computationally demanding. In contrast to ANNs, humans are not susceptible to adversarial attacks. Therefore, in this work, we investigate whether biologically-plausible learning algorithms are more robust against adversarial attacks than BP. In particular, we present an extensive comparative analysis of the adversarial robustness of BP and Present the Error to Perturb the Input To modulate Activity (PEPITA), a recently proposed biologically-plausible learning algorithm, on various computer vision tasks. We observe that PEPITA has higher intrinsic adversarial robustness and, with adversarial training, has a more favourable natural-vs-adversarial performance trade-off as, for the same natural accuracies, PEPITA's adversarial accuracies decrease in average by 0.26% and BP's by 8.05%.



## **8. An Integrated Algorithm for Robust and Imperceptible Audio Adversarial Examples**

cs.SD

Proc. 3rd Symposium on Security and Privacy in Speech Communication

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03349v1) [paper-pdf](http://arxiv.org/pdf/2310.03349v1)

**Authors**: Armin Ettenhofer, Jan-Philipp Schulze, Karla Pizzi

**Abstract**: Audio adversarial examples are audio files that have been manipulated to fool an automatic speech recognition (ASR) system, while still sounding benign to a human listener. Most methods to generate such samples are based on a two-step algorithm: first, a viable adversarial audio file is produced, then, this is fine-tuned with respect to perceptibility and robustness. In this work, we present an integrated algorithm that uses psychoacoustic models and room impulse responses (RIR) in the generation step. The RIRs are dynamically created by a neural network during the generation process to simulate a physical environment to harden our examples against transformations experienced in over-the-air attacks. We compare the different approaches in three experiments: in a simulated environment and in a realistic over-the-air scenario to evaluate the robustness, and in a human study to evaluate the perceptibility. Our algorithms considering psychoacoustics only or in addition to the robustness show an improvement in the signal-to-noise ratio (SNR) as well as in the human perception study, at the cost of an increased word error rate (WER).



## **9. Untargeted White-box Adversarial Attack with Heuristic Defence Methods in Real-time Deep Learning based Network Intrusion Detection System**

cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03334v1) [paper-pdf](http://arxiv.org/pdf/2310.03334v1)

**Authors**: Khushnaseeb Roshan, Aasim Zafar, Sheikh Burhan Ul Haque

**Abstract**: Network Intrusion Detection System (NIDS) is a key component in securing the computer network from various cyber security threats and network attacks. However, consider an unfortunate situation where the NIDS is itself attacked and vulnerable more specifically, we can say, How to defend the defender?. In Adversarial Machine Learning (AML), the malicious actors aim to fool the Machine Learning (ML) and Deep Learning (DL) models to produce incorrect predictions with intentionally crafted adversarial examples. These adversarial perturbed examples have become the biggest vulnerability of ML and DL based systems and are major obstacles to their adoption in real-time and mission-critical applications such as NIDS. AML is an emerging research domain, and it has become a necessity for the in-depth study of adversarial attacks and their defence strategies to safeguard the computer network from various cyber security threads. In this research work, we aim to cover important aspects related to NIDS, adversarial attacks and its defence mechanism to increase the robustness of the ML and DL based NIDS. We implemented four powerful adversarial attack techniques, namely, Fast Gradient Sign Method (FGSM), Jacobian Saliency Map Attack (JSMA), Projected Gradient Descent (PGD) and Carlini & Wagner (C&W) in NIDS. We analyzed its performance in terms of various performance metrics in detail. Furthermore, the three heuristics defence strategies, i.e., Adversarial Training (AT), Gaussian Data Augmentation (GDA) and High Confidence (HC), are implemented to improve the NIDS robustness under adversarial attack situations. The complete workflow is demonstrated in real-time network with data packet flow. This research work provides the overall background for the researchers interested in AML and its implementation from a computer network security point of view.



## **10. Certifiably Robust Graph Contrastive Learning**

cs.CR

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03312v1) [paper-pdf](http://arxiv.org/pdf/2310.03312v1)

**Authors**: Minhua Lin, Teng Xiao, Enyan Dai, Xiang Zhang, Suhang Wang

**Abstract**: Graph Contrastive Learning (GCL) has emerged as a popular unsupervised graph representation learning method. However, it has been shown that GCL is vulnerable to adversarial attacks on both the graph structure and node attributes. Although empirical approaches have been proposed to enhance the robustness of GCL, the certifiable robustness of GCL is still remain unexplored. In this paper, we develop the first certifiably robust framework in GCL. Specifically, we first propose a unified criteria to evaluate and certify the robustness of GCL. We then introduce a novel technique, RES (Randomized Edgedrop Smoothing), to ensure certifiable robustness for any GCL model, and this certified robustness can be provably preserved in downstream tasks. Furthermore, an effective training method is proposed for robust GCL. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed method in providing effective certifiable robustness and enhancing the robustness of any GCL model. The source code of RES is available at https://github.com/ventr1c/RES-GCL.



## **11. BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection**

cs.CR

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2308.12439v2) [paper-pdf](http://arxiv.org/pdf/2308.12439v2)

**Authors**: Tinghao Xie, Xiangyu Qi, Ping He, Yiming Li, Jiachen T. Wang, Prateek Mittal

**Abstract**: We present a novel defense, against backdoor attacks on Deep Neural Networks (DNNs), wherein adversaries covertly implant malicious behaviors (backdoors) into DNNs. Our defense falls within the category of post-development defenses that operate independently of how the model was generated. The proposed defense is built upon a novel reverse engineering approach that can directly extract backdoor functionality of a given backdoored model to a backdoor expert model. The approach is straightforward -- finetuning the backdoored model over a small set of intentionally mislabeled clean samples, such that it unlearns the normal functionality while still preserving the backdoor functionality, and thus resulting in a model (dubbed a backdoor expert model) that can only recognize backdoor inputs. Based on the extracted backdoor expert model, we show the feasibility of devising highly accurate backdoor input detectors that filter out the backdoor inputs during model inference. Further augmented by an ensemble strategy with a finetuned auxiliary model, our defense, BaDExpert (Backdoor Input Detection with Backdoor Expert), effectively mitigates 17 SOTA backdoor attacks while minimally impacting clean utility. The effectiveness of BaDExpert has been verified on multiple datasets (CIFAR10, GTSRB and ImageNet) across various model architectures (ResNet, VGG, MobileNetV2 and Vision Transformer).



## **12. Burning the Adversarial Bridges: Robust Windows Malware Detection Against Binary-level Mutations**

cs.LG

12 pages

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03285v1) [paper-pdf](http://arxiv.org/pdf/2310.03285v1)

**Authors**: Ahmed Abusnaina, Yizhen Wang, Sunpreet Arora, Ke Wang, Mihai Christodorescu, David Mohaisen

**Abstract**: Toward robust malware detection, we explore the attack surface of existing malware detection systems. We conduct root-cause analyses of the practical binary-level black-box adversarial malware examples. Additionally, we uncover the sensitivity of volatile features within the detection engines and exhibit their exploitability. Highlighting volatile information channels within the software, we introduce three software pre-processing steps to eliminate the attack surface, namely, padding removal, software stripping, and inter-section information resetting. Further, to counter the emerging section injection attacks, we propose a graph-based section-dependent information extraction scheme for software representation. The proposed scheme leverages aggregated information within various sections in the software to enable robust malware detection and mitigate adversarial settings. Our experimental results show that traditional malware detection models are ineffective against adversarial threats. However, the attack surface can be largely reduced by eliminating the volatile information. Therefore, we propose simple-yet-effective methods to mitigate the impacts of binary manipulation attacks. Overall, our graph-based malware detection scheme can accurately detect malware with an area under the curve score of 88.32\% and a score of 88.19% under a combination of binary manipulation attacks, exhibiting the efficiency of our proposed scheme.



## **13. Network Cascade Vulnerability using Constrained Bayesian Optimization**

cs.SI

13 pages, 5 figures

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2304.14420v2) [paper-pdf](http://arxiv.org/pdf/2304.14420v2)

**Authors**: Albert Lam, Mihai Anitescu, Anirudh Subramanyam

**Abstract**: Measures of power grid vulnerability are often assessed by the amount of damage an adversary can exact on the network. However, the cascading impact of such attacks is often overlooked, even though cascades are one of the primary causes of large-scale blackouts. This paper explores modifications of transmission line protection settings as candidates for adversarial attacks, which can remain undetectable as long as the network equilibrium state remains unaltered. This forms the basis of a black-box function in a Bayesian optimization procedure, where the objective is to find protection settings that maximize network degradation due to cascading. Notably, our proposed method is agnostic to the choice of the cascade simulator and its underlying assumptions. Numerical experiments reveal that, against conventional wisdom, maximally misconfiguring the protection settings of all network lines does not cause the most cascading. More surprisingly, even when the degree of misconfiguration is limited due to resource constraints, it is still possible to find settings that produce cascades comparable in severity to instances where there are no resource constraints.



## **14. LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference**

cs.LG

NeurIPS 2023 accepted publication

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2309.14331v3) [paper-pdf](http://arxiv.org/pdf/2309.14331v3)

**Authors**: Hongwu Peng, Ran Ran, Yukui Luo, Jiahui Zhao, Shaoyi Huang, Kiran Thorat, Tong Geng, Chenghong Wang, Xiaolin Xu, Wujie Wen, Caiwen Ding

**Abstract**: The growth of Graph Convolution Network (GCN) model sizes has revolutionized numerous applications, surpassing human performance in areas such as personal healthcare and financial systems. The deployment of GCNs in the cloud raises privacy concerns due to potential adversarial attacks on client data. To address security concerns, Privacy-Preserving Machine Learning (PPML) using Homomorphic Encryption (HE) secures sensitive client data. However, it introduces substantial computational overhead in practical applications. To tackle those challenges, we present LinGCN, a framework designed to reduce multiplication depth and optimize the performance of HE based GCN inference. LinGCN is structured around three key elements: (1) A differentiable structural linearization algorithm, complemented by a parameterized discrete indicator function, co-trained with model weights to meet the optimization goal. This strategy promotes fine-grained node-level non-linear location selection, resulting in a model with minimized multiplication depth. (2) A compact node-wise polynomial replacement policy with a second-order trainable activation function, steered towards superior convergence by a two-level distillation approach from an all-ReLU based teacher model. (3) an enhanced HE solution that enables finer-grained operator fusion for node-wise activation functions, further reducing multiplication level consumption in HE-based inference. Our experiments on the NTU-XVIEW skeleton joint dataset reveal that LinGCN excels in latency, accuracy, and scalability for homomorphically encrypted inference, outperforming solutions such as CryptoGCN. Remarkably, LinGCN achieves a 14.2x latency speedup relative to CryptoGCN, while preserving an inference accuracy of 75% and notably reducing multiplication depth.



## **15. Misusing Tools in Large Language Models With Visual Adversarial Examples**

cs.CR

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.03185v1) [paper-pdf](http://arxiv.org/pdf/2310.03185v1)

**Authors**: Xiaohan Fu, Zihan Wang, Shuheng Li, Rajesh K. Gupta, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Models (LLMs) are being enhanced with the ability to use tools and to process multiple modalities. These new capabilities bring new benefits and also new security risks. In this work, we show that an attacker can use visual adversarial examples to cause attacker-desired tool usage. For example, the attacker could cause a victim LLM to delete calendar events, leak private conversations and book hotels. Different from prior work, our attacks can affect the confidentiality and integrity of user resources connected to the LLM while being stealthy and generalizable to multiple input prompts. We construct these attacks using gradient-based adversarial training and characterize performance along multiple dimensions. We find that our adversarial images can manipulate the LLM to invoke tools following real-world syntax almost always (~98%) while maintaining high similarity to clean images (~0.9 SSIM). Furthermore, using human scoring and automated metrics, we find that the attacks do not noticeably affect the conversation (and its semantics) between the user and the LLM.



## **16. Raze to the Ground: Query-Efficient Adversarial HTML Attacks on Machine-Learning Phishing Webpage Detectors**

cs.CR

Proceedings of the 16th ACM Workshop on Artificial Intelligence and  Security (AISec '23), November 30, 2023, Copenhagen, Denmark

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.03166v1) [paper-pdf](http://arxiv.org/pdf/2310.03166v1)

**Authors**: Biagio Montaruli, Luca Demetrio, Maura Pintor, Luca Compagna, Davide Balzarotti, Battista Biggio

**Abstract**: Machine-learning phishing webpage detectors (ML-PWD) have been shown to suffer from adversarial manipulations of the HTML code of the input webpage. Nevertheless, the attacks recently proposed have demonstrated limited effectiveness due to their lack of optimizing the usage of the adopted manipulations, and they focus solely on specific elements of the HTML code. In this work, we overcome these limitations by first designing a novel set of fine-grained manipulations which allow to modify the HTML code of the input phishing webpage without compromising its maliciousness and visual appearance, i.e., the manipulations are functionality- and rendering-preserving by design. We then select which manipulations should be applied to bypass the target detector by a query-efficient black-box optimization algorithm. Our experiments show that our attacks are able to raze to the ground the performance of current state-of-the-art ML-PWD using just 30 queries, thus overcoming the weaker attacks developed in previous work, and enabling a much fairer robustness evaluation of ML-PWD.



## **17. LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples**

cs.CL

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.01469v2) [paper-pdf](http://arxiv.org/pdf/2310.01469v2)

**Authors**: Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan Ning, Li Yuan

**Abstract**: Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be knowledgeable and able to adapt to many tasks. However, we still can not completely trust their answer, since LLMs suffer from hallucination--fabricating non-existent facts to cheat users without perception. And the reasons for their existence and pervasiveness remain unclear. In this paper, we demonstrate that non-sense prompts composed of random tokens can also elicit the LLMs to respond with hallucinations. This phenomenon forces us to revisit that hallucination may be another view of adversarial examples, and it shares similar features with conventional adversarial examples as the basic feature of LLMs. Therefore, we formalize an automatic hallucination triggering method as the hallucination attack in an adversarial way. Finally, we explore basic feature of attacked adversarial prompts and propose a simple yet effective defense strategy. Our code is released on GitHub.



## **18. Optimizing Key-Selection for Face-based One-Time Biometrics via Morphing**

cs.CV

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.02997v1) [paper-pdf](http://arxiv.org/pdf/2310.02997v1)

**Authors**: Daile Osorio-Roig, Mahdi Ghafourian, Christian Rathgeb, Ruben Vera-Rodriguez, Christoph Busch, Julian Fierrez

**Abstract**: Nowadays, facial recognition systems are still vulnerable to adversarial attacks. These attacks vary from simple perturbations of the input image to modifying the parameters of the recognition model to impersonate an authorised subject. So-called privacy-enhancing facial recognition systems have been mostly developed to provide protection of stored biometric reference data, i.e. templates. In the literature, privacy-enhancing facial recognition approaches have focused solely on conventional security threats at the template level, ignoring the growing concern related to adversarial attacks. Up to now, few works have provided mechanisms to protect face recognition against adversarial attacks while maintaining high security at the template level. In this paper, we propose different key selection strategies to improve the security of a competitive cancelable scheme operating at the signal level. Experimental results show that certain strategies based on signal-level key selection can lead to complete blocking of the adversarial attack based on an iterative optimization for the most secure threshold, while for the most practical threshold, the attack success chance can be decreased to approximately 5.0%.



## **19. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

cs.CR

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2205.10159v4) [paper-pdf](http://arxiv.org/pdf/2205.10159v4)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstract**: Adversarial examples pose a security risk as they can alter decisions of a machine learning classifier through slight input perturbations. Certified robustness has been proposed as a mitigation where given an input $\mathbf{x}$, a classifier returns a prediction and a certified radius $R$ with a provable guarantee that any perturbation to $\mathbf{x}$ with $R$-bounded norm will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples against state-of-the-art certifications in two threat models, that differ in how the norm of the perturbation is computed. We show that the attack can be carried out against linear classifiers that have exact certifiable guarantees and against neural networks that have conservative certifications. In the weak threat model, our experiments demonstrate attack success rates over 50% on random linear classifiers, up to 23% on the MNIST dataset for linear SVM, and up to 15% for a neural network. In the strong threat model, the success rates are lower but positive. The floating-point errors exploited by our attacks can range from small to large (e.g., $10^{-13}$ to $10^{3}$) - showing that even negligible errors can be systematically exploited to invalidate guarantees provided by certified robustness. Finally, we propose a formal mitigation approach based on bounded interval arithmetic, encouraging future implementations of robustness certificates to account for limitations of modern computing architecture to provide sound certifiable guarantees.



## **20. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

cs.AI

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2309.10253v2) [paper-pdf](http://arxiv.org/pdf/2309.10253v2)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.



## **21. SlowFormer: Universal Adversarial Patch for Attack on Compute and Energy Efficiency of Inference Efficient Vision Transformers**

cs.CV

Code is available at https://github.com/UCDvision/SlowFormer

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.02544v1) [paper-pdf](http://arxiv.org/pdf/2310.02544v1)

**Authors**: KL Navaneet, Soroush Abbasi Koohpayegani, Essam Sleiman, Hamed Pirsiavash

**Abstract**: Recently, there has been a lot of progress in reducing the computation of deep models at inference time. These methods can reduce both the computational needs and power usage of deep models. Some of these approaches adaptively scale the compute based on the input instance. We show that such models can be vulnerable to a universal adversarial patch attack, where the attacker optimizes for a patch that when pasted on any image, can increase the compute and power consumption of the model. We run experiments with three different efficient vision transformer methods showing that in some cases, the attacker can increase the computation to the maximum possible level by simply pasting a patch that occupies only 8\% of the image area. We also show that a standard adversarial training defense method can reduce some of the attack's success. We believe adaptive efficient methods will be necessary for the future to lower the power usage of deep models, so we hope our paper encourages the community to study the robustness of these methods and develop better defense methods for the proposed attack.



## **22. Jailbreaker in Jail: Moving Target Defense for Large Language Models**

cs.CR

MTD Workshop in CCS'23

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02417v1) [paper-pdf](http://arxiv.org/pdf/2310.02417v1)

**Authors**: Bocheng Chen, Advait Paliwal, Qiben Yan

**Abstract**: Large language models (LLMs), known for their capability in understanding and following instructions, are vulnerable to adversarial attacks. Researchers have found that current commercial LLMs either fail to be "harmless" by presenting unethical answers, or fail to be "helpful" by refusing to offer meaningful answers when faced with adversarial queries. To strike a balance between being helpful and harmless, we design a moving target defense (MTD) enhanced LLM system. The system aims to deliver non-toxic answers that align with outputs from multiple model candidates, making them more robust against adversarial attacks. We design a query and output analysis model to filter out unsafe or non-responsive answers. %to achieve the two objectives of randomly selecting outputs from different LLMs. We evaluate over 8 most recent chatbot models with state-of-the-art adversarial queries. Our MTD-enhanced LLM system reduces the attack success rate from 37.5\% to 0\%. Meanwhile, it decreases the response refusal rate from 50\% to 0\%.



## **23. Exploring Model Learning Heterogeneity for Boosting Ensemble Robustness**

cs.CV

Accepted by IEEE ICDM 2023

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02237v1) [paper-pdf](http://arxiv.org/pdf/2310.02237v1)

**Authors**: Yanzhao Wu, Ka-Ho Chow, Wenqi Wei, Ling Liu

**Abstract**: Deep neural network ensembles hold the potential of improving generalization performance for complex learning tasks. This paper presents formal analysis and empirical evaluation to show that heterogeneous deep ensembles with high ensemble diversity can effectively leverage model learning heterogeneity to boost ensemble robustness. We first show that heterogeneous DNN models trained for solving the same learning problem, e.g., object detection, can significantly strengthen the mean average precision (mAP) through our weighted bounding box ensemble consensus method. Second, we further compose ensembles of heterogeneous models for solving different learning problems, e.g., object detection and semantic segmentation, by introducing the connected component labeling (CCL) based alignment. We show that this two-tier heterogeneity driven ensemble construction method can compose an ensemble team that promotes high ensemble diversity and low negative correlation among member models of the ensemble, strengthening ensemble robustness against both negative examples and adversarial attacks. Third, we provide a formal analysis of the ensemble robustness in terms of negative correlation. Extensive experiments validate the enhanced robustness of heterogeneous ensembles in both benign and adversarial settings. The source codes are available on GitHub at https://github.com/git-disl/HeteRobust.



## **24. Abusing Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs**

cs.CR

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2307.10490v4) [paper-pdf](http://arxiv.org/pdf/2307.10490v4)

**Authors**: Eugene Bagdasaryan, Tsung-Yin Hsieh, Ben Nassi, Vitaly Shmatikov

**Abstract**: We demonstrate how images and sounds can be used for indirect prompt and instruction injection in multi-modal LLMs. An attacker generates an adversarial perturbation corresponding to the prompt and blends it into an image or audio recording. When the user asks the (unmodified, benign) model about the perturbed image or audio, the perturbation steers the model to output the attacker-chosen text and/or make the subsequent dialog follow the attacker's instruction. We illustrate this attack with several proof-of-concept examples targeting LLaVa and PandaGPT.



## **25. FLEDGE: Ledger-based Federated Learning Resilient to Inference and Backdoor Attacks**

cs.CR

To appear in Annual Computer Security Applications Conference (ACSAC)  2023

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02113v1) [paper-pdf](http://arxiv.org/pdf/2310.02113v1)

**Authors**: Jorge Castillo, Phillip Rieger, Hossein Fereidooni, Qian Chen, Ahmad Sadeghi

**Abstract**: Federated learning (FL) is a distributed learning process that uses a trusted aggregation server to allow multiple parties (or clients) to collaboratively train a machine learning model without having them share their private data. Recent research, however, has demonstrated the effectiveness of inference and poisoning attacks on FL. Mitigating both attacks simultaneously is very challenging. State-of-the-art solutions have proposed the use of poisoning defenses with Secure Multi-Party Computation (SMPC) and/or Differential Privacy (DP). However, these techniques are not efficient and fail to address the malicious intent behind the attacks, i.e., adversaries (curious servers and/or compromised clients) seek to exploit a system for monetization purposes. To overcome these limitations, we present a ledger-based FL framework known as FLEDGE that allows making parties accountable for their behavior and achieve reasonable efficiency for mitigating inference and poisoning attacks. Our solution leverages crypto-currency to increase party accountability by penalizing malicious behavior and rewarding benign conduct. We conduct an extensive evaluation on four public datasets: Reddit, MNIST, Fashion-MNIST, and CIFAR-10. Our experimental results demonstrate that (1) FLEDGE provides strong privacy guarantees for model updates without sacrificing model utility; (2) FLEDGE can successfully mitigate different poisoning attacks without degrading the performance of the global model; and (3) FLEDGE offers unique reward mechanisms to promote benign behavior during model training and/or model aggregation.



## **26. Certifiers Make Neural Networks Vulnerable to Availability Attacks**

cs.LG

Published at 16th ACM Workshop on Artificial Intelligence and  Security (AISec '23)

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2108.11299v5) [paper-pdf](http://arxiv.org/pdf/2108.11299v5)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstract**: To achieve reliable, robust, and safe AI systems, it is vital to implement fallback strategies when AI predictions cannot be trusted. Certifiers for neural networks are a reliable way to check the robustness of these predictions. They guarantee for some predictions that a certain class of manipulations or attacks could not have changed the outcome. For the remaining predictions without guarantees, the method abstains from making a prediction, and a fallback strategy needs to be invoked, which typically incurs additional costs, can require a human operator, or even fail to provide any prediction. While this is a key concept towards safe and secure AI, we show for the first time that this approach comes with its own security risks, as such fallback strategies can be deliberately triggered by an adversary. In addition to naturally occurring abstains for some inputs and perturbations, the adversary can use training-time attacks to deliberately trigger the fallback with high probability. This transfers the main system load onto the fallback, reducing the overall system's integrity and/or availability. We design two novel availability attacks, which show the practical relevance of these threats. For example, adding 1% poisoned data during training is sufficient to trigger the fallback and hence make the model unavailable for up to 100% of all inputs by inserting the trigger. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the broad applicability of these attacks. An initial investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, specific solutions.



## **27. DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training**

cs.LG

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02025v1) [paper-pdf](http://arxiv.org/pdf/2310.02025v1)

**Authors**: Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu

**Abstract**: Zeroth-order (ZO) optimization has become a popular technique for solving machine learning (ML) problems when first-order (FO) information is difficult or impossible to obtain. However, the scalability of ZO optimization remains an open problem: Its use has primarily been limited to relatively small-scale ML problems, such as sample-wise adversarial attack generation. To our best knowledge, no prior work has demonstrated the effectiveness of ZO optimization in training deep neural networks (DNNs) without a significant decrease in performance. To overcome this roadblock, we develop DeepZero, a principled ZO deep learning (DL) framework that can scale ZO optimization to DNN training from scratch through three primary innovations. First, we demonstrate the advantages of coordinate-wise gradient estimation (CGE) over randomized vector-wise gradient estimation in training accuracy and computational efficiency. Second, we propose a sparsity-induced ZO training protocol that extends the model pruning methodology using only finite differences to explore and exploit the sparse DL prior in CGE. Third, we develop the methods of feature reuse and forward parallelization to advance the practical implementations of ZO training. Our extensive experiments show that DeepZero achieves state-of-the-art (SOTA) accuracy on ResNet-20 trained on CIFAR-10, approaching FO training performance for the first time. Furthermore, we show the practical utility of DeepZero in applications of certified adversarial defense and DL-based partial differential equation error correction, achieving 10-20% improvement over SOTA. We believe our results will inspire future research on scalable ZO optimization and contribute to advancing DL with black box.



## **28. Beyond Labeling Oracles: What does it mean to steal ML models?**

cs.LG

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.01959v1) [paper-pdf](http://arxiv.org/pdf/2310.01959v1)

**Authors**: Avital Shafran, Ilia Shumailov, Murat A. Erdogdu, Nicolas Papernot

**Abstract**: Model extraction attacks are designed to steal trained models with only query access, as is often provided through APIs that ML-as-a-Service providers offer. ML models are expensive to train, in part because data is hard to obtain, and a primary incentive for model extraction is to acquire a model while incurring less cost than training from scratch. Literature on model extraction commonly claims or presumes that the attacker is able to save on both data acquisition and labeling costs. We show that the attacker often does not. This is because current attacks implicitly rely on the adversary being able to sample from the victim model's data distribution. We thoroughly evaluate factors influencing the success of model extraction. We discover that prior knowledge of the attacker, i.e. access to in-distribution data, dominates other factors like the attack policy the adversary follows to choose which queries to make to the victim model API. Thus, an adversary looking to develop an equally capable model with a fixed budget has little practical incentive to perform model extraction, since for the attack to work they need to collect in-distribution data, saving only on the cost of labeling. With low labeling costs in the current market, the usefulness of such attacks is questionable. Ultimately, we demonstrate that the effect of prior knowledge needs to be explicitly decoupled from the attack policy. To this end, we propose a benchmark to evaluate attack policy directly.



## **29. Multi-Static ISAC in Cell-Free Massive MIMO: Precoder Design and Privacy Assessment**

eess.SP

Submitted to the 2023 IEEE Globecom Workshop on Enabling Security,  Trust, and Privacy in 6G Wireless Systems

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2309.13368v3) [paper-pdf](http://arxiv.org/pdf/2309.13368v3)

**Authors**: Isabella W. G. da Silva, Diana P. M. Osorio, Markku Juntti

**Abstract**: A multi-static sensing-centric integrated sensing and communication (ISAC) network can take advantage of the cell-free massive multiple-input multiple-output infrastructure to achieve remarkable diversity gains and reduced power consumption. While the conciliation of sensing and communication requirements is still a challenge, the privacy of the sensing information is a growing concern that should be seriously taken on the design of these systems to prevent other attacks. This paper tackles this issue by assessing the probability of an internal adversary to infer the target location information from the received signal by considering the design of transmit precoders that jointly optimizes the sensing and communication requirements in a multi-static-based cell-free ISAC network. Our results show that the multi-static setting facilitates a more precise estimation of the location of the target than the mono-static implementation.



## **30. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

cs.LG

Accepted in MM 2023

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2305.09241v5) [paper-pdf](http://arxiv.org/pdf/2305.09241v5)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning. Our code is available at \url{https://github.com/jiangw-0/LE_JCDP}.



## **31. DRSM: De-Randomized Smoothing on Malware Classifier Providing Certified Robustness**

cs.CR

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2303.13372v3) [paper-pdf](http://arxiv.org/pdf/2303.13372v3)

**Authors**: Shoumik Saha, Wenxiao Wang, Yigitcan Kaya, Soheil Feizi, Tudor Dumitras

**Abstract**: Machine Learning (ML) models have been utilized for malware detection for over two decades. Consequently, this ignited an ongoing arms race between malware authors and antivirus systems, compelling researchers to propose defenses for malware-detection models against evasion attacks. However, most if not all existing defenses against evasion attacks suffer from sizable performance degradation and/or can defend against only specific attacks, which makes them less practical in real-world settings. In this work, we develop a certified defense, DRSM (De-Randomized Smoothed MalConv), by redesigning the de-randomized smoothing technique for the domain of malware detection. Specifically, we propose a window ablation scheme to provably limit the impact of adversarial bytes while maximally preserving local structures of the executables. After showing how DRSM is theoretically robust against attacks with contiguous adversarial bytes, we verify its performance and certified robustness experimentally, where we observe only marginal accuracy drops as the cost of robustness. To our knowledge, we are the first to offer certified robustness in the realm of static detection of malware executables. More surprisingly, through evaluating DRSM against 9 empirical attacks of different types, we observe that the proposed defense is empirically robust to some extent against a diverse set of attacks, some of which even fall out of the scope of its original threat model. In addition, we collected 15.5K recent benign raw executables from diverse sources, which will be made public as a dataset called PACE (Publicly Accessible Collection(s) of Executables) to alleviate the scarcity of publicly available benign datasets for studying malware detection and provide future research with more representative data of the time.



## **32. Robust Offline Reinforcement Learning -- Certify the Confidence Interval**

cs.LG

the theoretical and experimental were only partial and incomplete

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2309.16631v2) [paper-pdf](http://arxiv.org/pdf/2309.16631v2)

**Authors**: Jiarui Yao, Simon Shaolei Du

**Abstract**: Currently, reinforcement learning (RL), especially deep RL, has received more and more attention in the research area. However, the security of RL has been an obvious problem due to the attack manners becoming mature. In order to defend against such adversarial attacks, several practical approaches are developed, such as adversarial training, data filtering, etc. However, these methods are mostly based on empirical algorithms and experiments, without rigorous theoretical analysis of the robustness of the algorithms. In this paper, we develop an algorithm to certify the robustness of a given policy offline with random smoothing, which could be proven and conducted as efficiently as ones without random smoothing. Experiments on different environments confirm the correctness of our algorithm.



## **33. Decision-Dominant Strategic Defense Against Lateral Movement for 5G Zero-Trust Multi-Domain Networks**

cs.CR

55 pages, 1 table, and 1 figures

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.01675v1) [paper-pdf](http://arxiv.org/pdf/2310.01675v1)

**Authors**: Tao Li, Yunian Pan, Quanyan Zhu

**Abstract**: Multi-domain warfare is a military doctrine that leverages capabilities from different domains, including air, land, sea, space, and cyberspace, to create a highly interconnected battle network that is difficult for adversaries to disrupt or defeat. However, the adoption of 5G technologies on battlefields presents new vulnerabilities due to the complexity of interconnections and the diversity of software, hardware, and devices from different supply chains. Therefore, establishing a zero-trust architecture for 5G-enabled networks is crucial for continuous monitoring and fast data analytics to protect against targeted attacks. To address these challenges, we propose a proactive end-to-end security scheme that utilizes a 5G satellite-guided air-ground network. Our approach incorporates a decision-dominant learning-based method that can thwart the lateral movement of adversaries targeting critical assets on the battlefield before they can conduct reconnaissance or gain necessary access or credentials. We demonstrate the effectiveness of our game-theoretic design, which uses a meta-learning framework to enable zero-trust monitoring and decision-dominant defense against attackers in emerging multi-domain battlefield networks.



## **34. Adversarial Client Detection via Non-parametric Subspace Monitoring in the Internet of Federated Things**

cs.LG

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.01537v1) [paper-pdf](http://arxiv.org/pdf/2310.01537v1)

**Authors**: Xianjian Xie, Xiaochen Xian, Dan Li, Andi Wang

**Abstract**: The Internet of Federated Things (IoFT) represents a network of interconnected systems with federated learning as the backbone, facilitating collaborative knowledge acquisition while ensuring data privacy for individual systems. The wide adoption of IoFT, however, is hindered by security concerns, particularly the susceptibility of federated learning networks to adversarial attacks. In this paper, we propose an effective non-parametric approach FedRR, which leverages the low-rank features of the transmitted parameter updates generated by federated learning to address the adversarial attack problem. Besides, our proposed method is capable of accurately detecting adversarial clients and controlling the false alarm rate under the scenario with no attack occurring. Experiments based on digit recognition using the MNIST datasets validated the advantages of our approach.



## **35. Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder**

cs.LG

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2303.15564v2) [paper-pdf](http://arxiv.org/pdf/2303.15564v2)

**Authors**: Tao Sun, Lu Pang, Chao Chen, Haibin Ling

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, where an adversary maliciously manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which are impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for black-box models. The true label of every test image needs to be recovered on the fly from a suspicious model regardless of image benignity. We focus on test-time image purification methods that incapacitate possible triggers while keeping semantic contents intact. Due to diverse trigger patterns and sizes, the heuristic trigger search in image space can be unscalable. We circumvent such barrier by leveraging the strong reconstruction power of generative models, and propose a framework of Blind Defense with Masked AutoEncoder (BDMAE). It detects possible triggers in the token space using image structural similarity and label consistency between the test image and MAE restorations. The detection results are then refined by considering trigger topology. Finally, we fuse MAE restorations adaptively into a purified image for making prediction. Our approach is blind to the model architectures, trigger patterns and image benignity. Extensive experiments under different backdoor settings validate its effectiveness and generalizability. Code is available at https://github.com/tsun/BDMAE.



## **36. Distributed Energy Resources Cybersecurity Outlook: Vulnerabilities, Attacks, Impacts, and Mitigations**

cs.CR

IEEE Systems Journal (2023)

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2205.11171v4) [paper-pdf](http://arxiv.org/pdf/2205.11171v4)

**Authors**: Ioannis Zografopoulos, Nikos D. Hatziargyriou, Charalambos Konstantinou

**Abstract**: The digitization and decentralization of the electric power grid are key thrusts for an economically and environmentally sustainable future. Towards this goal, distributed energy resources (DER), including rooftop solar panels, battery storage, electric vehicles, etc., are becoming ubiquitous in power systems. Power utilities benefit from DERs as they minimize operational costs; at the same time, DERs grant users and aggregators control over the power they produce and consume. DERs are interconnected, interoperable, and support remotely controllable features, thus, their cybersecurity is of cardinal importance. DER communication dependencies and the diversity of DER architectures widen the threat surface and aggravate the cybersecurity posture of power systems. In this work, we focus on security oversights that reside in the cyber and physical layers of DERs and can jeopardize grid operations. Existing works have underlined the impact of cyberattacks targeting DER assets, however, they either focus on specific system components (e.g., communication protocols), do not consider the mission-critical objectives of DERs, or neglect the adversarial perspective (e.g., adversary/attack models) altogether. To address these omissions, we comprehensively analyze adversarial capabilities and objectives when manipulating DER assets, and then present how protocol and device-level vulnerabilities can materialize into cyberattacks impacting power system operations. Finally, we provide mitigation strategies to thwart adversaries and directions for future DER cybersecurity research.



## **37. Fooling the Textual Fooler via Randomizing Latent Representations**

cs.CL

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.01452v1) [paper-pdf](http://arxiv.org/pdf/2310.01452v1)

**Authors**: Duy C. Hoang, Quang H. Nguyen, Saurav Manchanda, MinLong Peng, Kok-Seng Wong, Khoa D. Doan

**Abstract**: Despite outstanding performance in a variety of NLP tasks, recent studies have revealed that NLP models are vulnerable to adversarial attacks that slightly perturb the input to cause the models to misbehave. Among these attacks, adversarial word-level perturbations are well-studied and effective attack strategies. Since these attacks work in black-box settings, they do not require access to the model architecture or model parameters and thus can be detrimental to existing NLP applications. To perform an attack, the adversary queries the victim model many times to determine the most important words in an input text and to replace these words with their corresponding synonyms. In this work, we propose a lightweight and attack-agnostic defense whose main goal is to perplex the process of generating an adversarial example in these query-based black-box attacks; that is to fool the textual fooler. This defense, named AdvFooler, works by randomizing the latent representation of the input at inference time. Different from existing defenses, AdvFooler does not necessitate additional computational overhead during training nor relies on assumptions about the potential adversarial perturbation set while having a negligible impact on the model's accuracy. Our theoretical and empirical analyses highlight the significance of robustness resulting from confusing the adversary via randomizing the latent space, as well as the impact of randomization on clean accuracy. Finally, we empirically demonstrate near state-of-the-art robustness of AdvFooler against representative adversarial word-level attacks on two benchmark datasets.



## **38. Counterfactual Image Generation for adversarially robust and interpretable Classifiers**

cs.CV

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00761v1) [paper-pdf](http://arxiv.org/pdf/2310.00761v1)

**Authors**: Rafael Bischof, Florian Scheidegger, Michael A. Kraus, A. Cristiano I. Malossi

**Abstract**: Neural Image Classifiers are effective but inherently hard to interpret and susceptible to adversarial attacks. Solutions to both problems exist, among others, in the form of counterfactual examples generation to enhance explainability or adversarially augment training datasets for improved robustness. However, existing methods exclusively address only one of the issues. We propose a unified framework leveraging image-to-image translation Generative Adversarial Networks (GANs) to produce counterfactual samples that highlight salient regions for interpretability and act as adversarial samples to augment the dataset for more robustness. This is achieved by combining the classifier and discriminator into a single model that attributes real images to their respective classes and flags generated images as "fake". We assess the method's effectiveness by evaluating (i) the produced explainability masks on a semantic segmentation task for concrete cracks and (ii) the model's resilience against the Projected Gradient Descent (PGD) attack on a fruit defects detection problem. Our produced saliency maps are highly descriptive, achieving competitive IoU values compared to classical segmentation models despite being trained exclusively on classification labels. Furthermore, the model exhibits improved robustness to adversarial attacks, and we show how the discriminator's "fakeness" value serves as an uncertainty measure of the predictions.



## **39. Silent Killer: A Stealthy, Clean-Label, Black-Box Backdoor Attack**

cs.CR

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2301.02615v2) [paper-pdf](http://arxiv.org/pdf/2301.02615v2)

**Authors**: Tzvi Lederer, Gallil Maimon, Lior Rokach

**Abstract**: Backdoor poisoning attacks pose a well-known risk to neural networks. However, most studies have focused on lenient threat models. We introduce Silent Killer, a novel attack that operates in clean-label, black-box settings, uses a stealthy poison and trigger and outperforms existing methods. We investigate the use of universal adversarial perturbations as triggers in clean-label attacks, following the success of such approaches under poison-label settings. We analyze the success of a naive adaptation and find that gradient alignment for crafting the poison is required to ensure high success rates. We conduct thorough experiments on MNIST, CIFAR10, and a reduced version of ImageNet and achieve state-of-the-art results.



## **40. DISCO Might Not Be Funky: Random Intelligent Reflective Surface Configurations That Attack**

eess.SP

This work has been submitted for possible publication. We will share  the main codes of this work via GitHub soon. arXiv admin note: text overlap  with arXiv:2308.15716

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00687v1) [paper-pdf](http://arxiv.org/pdf/2310.00687v1)

**Authors**: Huan Huang, Lipeng Dai, Hongliang Zhang, Chongfu Zhang, Zhongxing Tian, Yi Cai, A. Lee Swindlehurst, Zhu Han

**Abstract**: Emerging intelligent reflective surfaces (IRSs) significantly improve system performance, but also pose a signifcant risk for physical layer security (PLS). Unlike the extensive research on legitimate IRS-enhanced communications, in this article we present an adversarial IRS-based fully-passive jammer (FPJ). We describe typical application scenarios for Disco IRS (DIRS)-based FPJ, where an illegitimate IRS with random, time-varying reflection properties acts like a "disco ball" to randomly change the propagation environment. We introduce the principles of DIRS-based FPJ and overview existing investigations of the technology, including a design example employing one-bit phase shifters. The DIRS-based FPJ can be implemented without either jamming power or channel state information (CSI) for the legitimate users (LUs). It does not suffer from the energy constraints of traditional active jammers, nor does it require any knowledge of the LU channels. In addition to the proposed jamming attack, we also propose an anti-jamming strategy that requires only statistical rather than instantaneous CSI. Furthermore, we present a data frame structure that enables the legitimate access point (AP) to estimate the statistical CSI in the presence of the DIRS jamming. Typical cases are discussed to show the impact of the DIRS-based FPJ and the feasibility of the anti-jamming precoder. Moreover, we outline future research directions and challenges for the DIRS-based FPJ and its anti-jamming precoding to stimulate this line of research and pave the way for practical applications.



## **41. A Survey of Robustness and Safety of 2D and 3D Deep Learning Models Against Adversarial Attacks**

cs.LG

Submitted to CSUR

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00633v1) [paper-pdf](http://arxiv.org/pdf/2310.00633v1)

**Authors**: Yanjie Li, Bin Xie, Songtao Guo, Yuanyuan Yang, Bin Xiao

**Abstract**: Benefiting from the rapid development of deep learning, 2D and 3D computer vision applications are deployed in many safe-critical systems, such as autopilot and identity authentication. However, deep learning models are not trustworthy enough because of their limited robustness against adversarial attacks. The physically realizable adversarial attacks further pose fatal threats to the application and human safety. Lots of papers have emerged to investigate the robustness and safety of deep learning models against adversarial attacks. To lead to trustworthy AI, we first construct a general threat model from different perspectives and then comprehensively review the latest progress of both 2D and 3D adversarial attacks. We extend the concept of adversarial examples beyond imperceptive perturbations and collate over 170 papers to give an overview of deep learning model robustness against various adversarial attacks. To the best of our knowledge, we are the first to systematically investigate adversarial attacks for 3D models, a flourishing field applied to many real-world applications. In addition, we examine physical adversarial attacks that lead to safety violations. Last but not least, we summarize present popular topics, give insights on challenges, and shed light on future research on trustworthy AI.



## **42. Generating Transferable and Stealthy Adversarial Patch via Attention-guided Adversarial Inpainting**

cs.CV

Submitted to ICLR2024

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2308.05320v2) [paper-pdf](http://arxiv.org/pdf/2308.05320v2)

**Authors**: Yanjie Li, Mingxing Duan, Xuelong Dai, Bin Xiao

**Abstract**: Adversarial patch attacks can fool the face recognition (FR) models via small patches. However, previous adversarial patch attacks often result in unnatural patterns that are easily noticeable. Generating transferable and stealthy adversarial patches that can efficiently deceive the black-box FR models while having good camouflage is challenging because of the huge stylistic difference between the source and target images. To generate transferable, natural-looking, and stealthy adversarial patches, we propose an innovative two-stage attack called Adv-Inpainting, which extracts style features and identity features from the attacker and target faces, respectively and then fills the patches with misleading and inconspicuous content guided by attention maps. In the first stage, we extract multi-scale style embeddings by a pyramid-like network and identity embeddings by a pretrained FR model and propose a novel Attention-guided Adaptive Instance Normalization layer (AAIN) to merge them via background-patch cross-attention maps. The proposed layer can adaptively fuse identity and style embeddings by fully exploiting priority contextual information. In the second stage, we design an Adversarial Patch Refinement Network (APR-Net) with a novel boundary variance loss, a spatial discounted reconstruction loss, and a perceptual loss to boost the stealthiness further. Experiments demonstrate that our attack can generate adversarial patches with improved visual quality, better stealthiness, and stronger transferability than state-of-the-art adversarial patch attacks and semantic attacks.



## **43. Understanding Adversarial Transferability in Federated Learning**

cs.LG

10 pages of the main paper. 21 pages in total

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00616v1) [paper-pdf](http://arxiv.org/pdf/2310.00616v1)

**Authors**: Yijiang Li, Ying Gao, Haohan Wang

**Abstract**: We investigate the robustness and security issues from a novel and practical setting: a group of malicious clients has impacted the model during training by disguising their identities and acting as benign clients, and only revealing their adversary position after the training to conduct transferable adversarial attacks with their data, which is usually a subset of the data that FL system is trained with. Our aim is to offer a full understanding of the challenges the FL system faces in this practical setting across a spectrum of configurations. We notice that such an attack is possible, but the federated model is more robust compared with its centralized counterpart when the accuracy on clean images is comparable. Through our study, we hypothesized the robustness is from two factors: the decentralized training on distributed data and the averaging operation. We provide evidence from both the perspective of empirical experiments and theoretical analysis. Our work has implications for understanding the robustness of federated learning systems and poses a practical question for federated learning applications.



## **44. On the Onset of Robust Overfitting in Adversarial Training**

cs.LG

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00607v1) [paper-pdf](http://arxiv.org/pdf/2310.00607v1)

**Authors**: Chaojian Yu, Xiaolong Shi, Jun Yu, Bo Han, Tongliang Liu

**Abstract**: Adversarial Training (AT) is a widely-used algorithm for building robust neural networks, but it suffers from the issue of robust overfitting, the fundamental mechanism of which remains unclear. In this work, we consider normal data and adversarial perturbation as separate factors, and identify that the underlying causes of robust overfitting stem from the normal data through factor ablation in AT. Furthermore, we explain the onset of robust overfitting as a result of the model learning features that lack robust generalization, which we refer to as non-effective features. Specifically, we provide a detailed analysis of the generation of non-effective features and how they lead to robust overfitting. Additionally, we explain various empirical behaviors observed in robust overfitting and revisit different techniques to mitigate robust overfitting from the perspective of non-effective features, providing a comprehensive understanding of the robust overfitting phenomenon. This understanding inspires us to propose two measures, attack strength and data augmentation, to hinder the learning of non-effective features by the neural network, thereby alleviating robust overfitting. Extensive experiments conducted on benchmark datasets demonstrate the effectiveness of the proposed methods in mitigating robust overfitting and enhancing adversarial robustness.



## **45. Physical Adversarial Attack meets Computer Vision: A Decade Survey**

cs.CV

19 pages. Under Review

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2209.15179v3) [paper-pdf](http://arxiv.org/pdf/2209.15179v3)

**Authors**: Hui Wei, Hao Tang, Xuemei Jia, Zhixiang Wang, Hanxun Yu, Zhubo Li, Shin'ichi Satoh, Luc Van Gool, Zheng Wang

**Abstract**: Despite the impressive achievements of Deep Neural Networks (DNNs) in computer vision, their vulnerability to adversarial attacks remains a critical concern. Extensive research has demonstrated that incorporating sophisticated perturbations into input images can lead to a catastrophic degradation in DNNs' performance. This perplexing phenomenon not only exists in the digital space but also in the physical world. Consequently, it becomes imperative to evaluate the security of DNNs-based systems to ensure their safe deployment in real-world scenarios, particularly in security-sensitive applications. To facilitate a profound understanding of this topic, this paper presents a comprehensive overview of physical adversarial attacks. Firstly, we distill four general steps for launching physical adversarial attacks. Building upon this foundation, we uncover the pervasive role of artifacts carrying adversarial perturbations in the physical world. These artifacts influence each step. To denote them, we introduce a new term: adversarial medium. Then, we take the first step to systematically evaluate the performance of physical adversarial attacks, taking the adversarial medium as a first attempt. Our proposed evaluation metric, hiPAA, comprises six perspectives: Effectiveness, Stealthiness, Robustness, Practicability, Aesthetics, and Economics. We also provide comparative results across task categories, together with insightful observations and suggestions for future research directions.



## **46. Understanding the Robustness of Randomized Feature Defense Against Query-Based Adversarial Attacks**

cs.LG

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00567v1) [paper-pdf](http://arxiv.org/pdf/2310.00567v1)

**Authors**: Quang H. Nguyen, Yingjie Lao, Tung Pham, Kok-Seng Wong, Khoa D. Doan

**Abstract**: Recent works have shown that deep neural networks are vulnerable to adversarial examples that find samples close to the original image but can make the model misclassify. Even with access only to the model's output, an attacker can employ black-box attacks to generate such adversarial examples. In this work, we propose a simple and lightweight defense against black-box attacks by adding random noise to hidden features at intermediate layers of the model at inference time. Our theoretical analysis confirms that this method effectively enhances the model's resilience against both score-based and decision-based black-box attacks. Importantly, our defense does not necessitate adversarial training and has minimal impact on accuracy, rendering it applicable to any pre-trained model. Our analysis also reveals the significance of selectively adding noise to different parts of the model based on the gradient of the adversarial objective function, which can be varied during the attack. We demonstrate the robustness of our defense against multiple black-box attacks through extensive empirical experiments involving diverse models with various architectures.



## **47. Black-box Attacks on Image Activity Prediction and its Natural Language Explanations**

cs.CV

Accepted at ICCV2023 AROW Workshop

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2310.00503v1) [paper-pdf](http://arxiv.org/pdf/2310.00503v1)

**Authors**: Alina Elena Baia, Valentina Poggioni, Andrea Cavallaro

**Abstract**: Explainable AI (XAI) methods aim to describe the decision process of deep neural networks. Early XAI methods produced visual explanations, whereas more recent techniques generate multimodal explanations that include textual information and visual representations. Visual XAI methods have been shown to be vulnerable to white-box and gray-box adversarial attacks, with an attacker having full or partial knowledge of and access to the target system. As the vulnerabilities of multimodal XAI models have not been examined, in this paper we assess for the first time the robustness to black-box attacks of the natural language explanations generated by a self-rationalizing image-based activity recognition model. We generate unrestricted, spatially variant perturbations that disrupt the association between the predictions and the corresponding explanations to mislead the model into generating unfaithful explanations. We show that we can create adversarial images that manipulate the explanations of an activity recognition model by having access only to its final output.



## **48. Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection**

cs.CL

The data and code can be found at  https://github.com/Leezekun/Adv-Instruct-Eval

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2308.10819v2) [paper-pdf](http://arxiv.org/pdf/2308.10819v2)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in following instructions, making them valuable in customer-facing applications. However, their impressive capabilities also raise concerns about the amplification of risks posed by adversarial instructions, which can be injected into the model input by third-party attackers to manipulate LLMs' original instructions and prompt unintended actions and content. Therefore, it is crucial to understand LLMs' ability to accurately discern which instructions to follow to ensure their safe deployment in real-world scenarios. In this paper, we propose a pioneering benchmark for automatically evaluating the robustness of instruction-following LLMs against adversarial instructions injected in the prompt. The objective of this benchmark is to quantify the extent to which LLMs are influenced by injected adversarial instructions and assess their ability to differentiate between these injected adversarial instructions and original user instructions. Through experiments conducted with state-of-the-art instruction-following LLMs, we uncover significant limitations in their robustness against adversarial instruction injection attacks. Furthermore, our findings indicate that prevalent instruction-tuned models are prone to being ``overfitted'' to follow any instruction phrase in the prompt without truly understanding which instructions should be followed. This highlights the need to address the challenge of training models to comprehend prompts instead of merely following instruction phrases and completing the text. The data and code can be found at \url{https://github.com/Leezekun/Adv-Instruct-Eval}.



## **49. Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems**

cs.LG

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2303.12981v3) [paper-pdf](http://arxiv.org/pdf/2303.12981v3)

**Authors**: Sihan Zeng, Thinh T. Doan, Justin Romberg

**Abstract**: The aim of this paper is to improve the understanding of the optimization landscape for policy optimization problems in reinforcement learning. Specifically, we show that the superlevel set of the objective function with respect to the policy parameter is always a connected set both in the tabular setting and under policies represented by a class of neural networks. In addition, we show that the optimization objective as a function of the policy parameter and reward satisfies a stronger "equiconnectedness" property. To our best knowledge, these are novel and previously unknown discoveries.   We present an application of the connectedness of these superlevel sets to the derivation of minimax theorems for robust reinforcement learning. We show that any minimax optimization program which is convex on one side and is equiconnected on the other side observes the minimax equality (i.e. has a Nash equilibrium). We find that this exact structure is exhibited by an interesting robust reinforcement learning problem under an adversarial reward attack, and the validity of its minimax equality immediately follows. This is the first time such a result is established in the literature.



## **50. Human-Producible Adversarial Examples**

cs.CV

Submitted to ICLR 2024

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2310.00438v1) [paper-pdf](http://arxiv.org/pdf/2310.00438v1)

**Authors**: David Khachaturov, Yue Gao, Ilia Shumailov, Robert Mullins, Ross Anderson, Kassem Fawaz

**Abstract**: Visual adversarial examples have so far been restricted to pixel-level image manipulations in the digital world, or have required sophisticated equipment such as 2D or 3D printers to be produced in the physical real world. We present the first ever method of generating human-producible adversarial examples for the real world that requires nothing more complicated than a marker pen. We call them $\textbf{adversarial tags}$. First, building on top of differential rendering, we demonstrate that it is possible to build potent adversarial examples with just lines. We find that by drawing just $4$ lines we can disrupt a YOLO-based model in $54.8\%$ of cases; increasing this to $9$ lines disrupts $81.8\%$ of the cases tested. Next, we devise an improved method for line placement to be invariant to human drawing error. We evaluate our system thoroughly in both digital and analogue worlds and demonstrate that our tags can be applied by untrained humans. We demonstrate the effectiveness of our method for producing real-world adversarial examples by conducting a user study where participants were asked to draw over printed images using digital equivalents as guides. We further evaluate the effectiveness of both targeted and untargeted attacks, and discuss various trade-offs and method limitations, as well as the practical and ethical implications of our work. The source code will be released publicly.



