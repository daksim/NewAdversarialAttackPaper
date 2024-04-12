# Latest Adversarial Attack Papers
**update at 2024-04-12 09:24:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

cs.CL

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07921v1) [paper-pdf](http://arxiv.org/pdf/2404.07921v1)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.



## **2. A Measurement of Genuine Tor Traces for Realistic Website Fingerprinting**

cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07892v1) [paper-pdf](http://arxiv.org/pdf/2404.07892v1)

**Authors**: Rob Jansen, Ryan Wails, Aaron Johnson

**Abstract**: Website fingerprinting (WF) is a dangerous attack on web privacy because it enables an adversary to predict the website a user is visiting, despite the use of encryption, VPNs, or anonymizing networks such as Tor. Previous WF work almost exclusively uses synthetic datasets to evaluate the performance and estimate the feasibility of WF attacks despite evidence that synthetic data misrepresents the real world. In this paper we present GTT23, the first WF dataset of genuine Tor traces, which we obtain through a large-scale measurement of the Tor network. GTT23 represents real Tor user behavior better than any existing WF dataset, is larger than any existing WF dataset by at least an order of magnitude, and will help ground the future study of realistic WF attacks and defenses. In a detailed evaluation, we survey 25 WF datasets published over the last 15 years and compare their characteristics to those of GTT23. We discover common deficiencies of synthetic datasets that make them inferior to GTT23 for drawing meaningful conclusions about the effectiveness of WF attacks directed at real Tor users. We have made GTT23 available to promote reproducible research and to help inspire new directions for future work.



## **3. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

cs.RO

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07880v1) [paper-pdf](http://arxiv.org/pdf/2404.07880v1)

**Authors**: Jiazhen Li, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.



## **4. LeapFrog: The Rowhammer Instruction Skip Attack**

cs.CR

Accepted at Hardware.io 2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07878v1) [paper-pdf](http://arxiv.org/pdf/2404.07878v1)

**Authors**: Andrew Adiletta, Caner Tol, Berk Sunar

**Abstract**: Since its inception, Rowhammer exploits have rapidly evolved into increasingly sophisticated threats not only compromising data integrity but also the control flow integrity of victim processes. Nevertheless, it remains a challenge for an attacker to identify vulnerable targets (i.e., Rowhammer gadgets), understand the outcome of the attempted fault, and formulate an attack that yields useful results.   In this paper, we present a new type of Rowhammer gadget, called a LeapFrog gadget, which, when present in the victim code, allows an adversary to subvert code execution to bypass a critical piece of code (e.g., authentication check logic, encryption rounds, padding in security protocols). The Leapfrog gadget manifests when the victim code stores the Program Counter (PC) value in the user or kernel stack (e.g., a return address during a function call) which, when tampered with, re-positions the return address to a location that bypasses a security-critical code pattern.   This research also presents a systematic process to identify Leapfrog gadgets. This methodology enables the automated detection of susceptible targets and the determination of optimal attack parameters. We first showcase this new attack vector through a practical demonstration on a TLS handshake client/server scenario, successfully inducing an instruction skip in a client application. We then demonstrate the attack on real-world code found in the wild, implementing an attack on OpenSSL.   Our findings extend the impact of Rowhammer attacks on control flow and contribute to the development of more robust defenses against these increasingly sophisticated threats.



## **5. Pilot Spoofing Attack on the Downlink of Cell-Free Massive MIMO: From the Perspective of Adversaries**

cs.IT

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2403.04435v2) [paper-pdf](http://arxiv.org/pdf/2403.04435v2)

**Authors**: Weiyang Xu, Ruiguang Wang, Yuan Zhang, Hien Quoc Ngo, Wei Xiang

**Abstract**: The channel hardening effect is less pronounced in the cell-free massive multiple-input multiple-output (mMIMO) system compared to its cellular counterpart, making it necessary to estimate the downlink effective channel gains to ensure decent performance. However, the downlink training inadvertently creates an opportunity for adversarial nodes to launch pilot spoofing attacks (PSAs). First, we demonstrate that adversarial distributed access points (APs) can severely degrade the achievable downlink rate. They achieve this by estimating their channels to users in the uplink training phase and then precoding and sending the same pilot sequences as those used by legitimate APs during the downlink training phase. Then, the impact of the downlink PSA is investigated by rigorously deriving a closed-form expression of the per-user achievable downlink rate. By employing the min-max criterion to optimize the power allocation coefficients, the maximum per-user achievable rate of downlink transmission is minimized from the perspective of adversarial APs. As an alternative to the downlink PSA, adversarial APs may opt to precode random interference during the downlink data transmission phase in order to disrupt legitimate communications. In this scenario, the achievable downlink rate is derived, and then power optimization algorithms are also developed. We present numerical results to showcase the detrimental impact of the downlink PSA and compare the effects of these two types of attacks.



## **6. Poisoning Prevention in Federated Learning and Differential Privacy via Stateful Proofs of Execution**

cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.06721v2) [paper-pdf](http://arxiv.org/pdf/2404.06721v2)

**Authors**: Norrathep Rattanavipanon, Ivan De Oliveira Nunes

**Abstract**: The rise in IoT-driven distributed data analytics, coupled with increasing privacy concerns, has led to a demand for effective privacy-preserving and federated data collection/model training mechanisms. In response, approaches such as Federated Learning (FL) and Local Differential Privacy (LDP) have been proposed and attracted much attention over the past few years. However, they still share the common limitation of being vulnerable to poisoning attacks wherein adversaries compromising edge devices feed forged (a.k.a. poisoned) data to aggregation back-ends, undermining the integrity of FL/LDP results.   In this work, we propose a system-level approach to remedy this issue based on a novel security notion of Proofs of Stateful Execution (PoSX) for IoT/embedded devices' software. To realize the PoSX concept, we design SLAPP: a System-Level Approach for Poisoning Prevention. SLAPP leverages commodity security features of embedded devices - in particular ARM TrustZoneM security extensions - to verifiably bind raw sensed data to their correct usage as part of FL/LDP edge device routines. As a consequence, it offers robust security guarantees against poisoning. Our evaluation, based on real-world prototypes featuring multiple cryptographic primitives and data collection schemes, showcases SLAPP's security and low overhead.



## **7. Enhancing Network Intrusion Detection Performance using Generative Adversarial Networks**

cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07464v1) [paper-pdf](http://arxiv.org/pdf/2404.07464v1)

**Authors**: Xinxing Zhao, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Network intrusion detection systems (NIDS) play a pivotal role in safeguarding critical digital infrastructures against cyber threats. Machine learning-based detection models applied in NIDS are prevalent today. However, the effectiveness of these machine learning-based models is often limited by the evolving and sophisticated nature of intrusion techniques as well as the lack of diverse and updated training samples. In this research, a novel approach for enhancing the performance of an NIDS through the integration of Generative Adversarial Networks (GANs) is proposed. By harnessing the power of GANs in generating synthetic network traffic data that closely mimics real-world network behavior, we address a key challenge associated with NIDS training datasets, which is the data scarcity. Three distinct GAN models (Vanilla GAN, Wasserstein GAN and Conditional Tabular GAN) are implemented in this work to generate authentic network traffic patterns specifically tailored to represent the anomalous activity. We demonstrate how this synthetic data resampling technique can significantly improve the performance of the NIDS model for detecting such activity. By conducting comprehensive experiments using the CIC-IDS2017 benchmark dataset, augmented with GAN-generated data, we offer empirical evidence that shows the effectiveness of our proposed approach. Our findings show that the integration of GANs into NIDS can lead to enhancements in intrusion detection performance for attacks with limited training data, making it a promising avenue for bolstering the cybersecurity posture of organizations in an increasingly interconnected and vulnerable digital landscape.



## **8. Privacy preserving layer partitioning for Deep Neural Network models**

cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07437v1) [paper-pdf](http://arxiv.org/pdf/2404.07437v1)

**Authors**: Kishore Rajasekar, Randolph Loh, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: MLaaS (Machine Learning as a Service) has become popular in the cloud computing domain, allowing users to leverage cloud resources for running private inference of ML models on their data. However, ensuring user input privacy and secure inference execution is essential. One of the approaches to protect data privacy and integrity is to use Trusted Execution Environments (TEEs) by enabling execution of programs in secure hardware enclave. Using TEEs can introduce significant performance overhead due to the additional layers of encryption, decryption, security and integrity checks. This can lead to slower inference times compared to running on unprotected hardware. In our work, we enhance the runtime performance of ML models by introducing layer partitioning technique and offloading computations to GPU. The technique comprises two distinct partitions: one executed within the TEE, and the other carried out using a GPU accelerator. Layer partitioning exposes intermediate feature maps in the clear which can lead to reconstruction attacks to recover the input. We conduct experiments to demonstrate the effectiveness of our approach in protecting against input reconstruction attacks developed using trained conditional Generative Adversarial Network(c-GAN). The evaluation is performed on widely used models such as VGG-16, ResNet-50, and EfficientNetB0, using two datasets: ImageNet for Image classification and TON IoT dataset for cybersecurity attack detection.



## **9. Multi-granular Adversarial Attacks against Black-box Neural Ranking Models**

cs.IR

Accepted by SIGIR2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.01574v2) [paper-pdf](http://arxiv.org/pdf/2404.01574v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Adversarial ranking attacks have gained increasing attention due to their success in probing vulnerabilities, and, hence, enhancing the robustness, of neural ranking models. Conventional attack methods employ perturbations at a single granularity, e.g., word or sentence level, to target documents. However, limiting perturbations to a single level of granularity may reduce the flexibility of adversarial examples, thereby diminishing the potential threat of the attack. Therefore, we focus on generating high-quality adversarial examples by incorporating multi-granular perturbations. Achieving this objective involves tackling a combinatorial explosion problem, which requires identifying an optimal combination of perturbations across all possible levels of granularity, positions, and textual pieces. To address this challenge, we transform the multi-granular adversarial attack into a sequential decision-making process, where perturbations in the next attack step build on the perturbed document in the current attack step. Since the attack process can only access the final state without direct intermediate signals, we use reinforcement learning to perform multi-granular attacks. During the reinforcement learning process, two agents work cooperatively to identify multi-granular vulnerabilities as attack targets and organize perturbation candidates into a final perturbation sequence. Experimental results show that our attack method surpasses prevailing baselines in both attack effectiveness and imperceptibility.



## **10. Incremental Randomized Smoothing Certification**

cs.LG

ICLR 2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2305.19521v2) [paper-pdf](http://arxiv.org/pdf/2305.19521v2)

**Authors**: Shubham Ugare, Tarun Suresh, Debangshu Banerjee, Gagandeep Singh, Sasa Misailovic

**Abstract**: Randomized smoothing-based certification is an effective approach for obtaining robustness certificates of deep neural networks (DNNs) against adversarial attacks. This method constructs a smoothed DNN model and certifies its robustness through statistical sampling, but it is computationally expensive, especially when certifying with a large number of samples. Furthermore, when the smoothed model is modified (e.g., quantized or pruned), certification guarantees may not hold for the modified DNN, and recertifying from scratch can be prohibitively expensive.   We present the first approach for incremental robustness certification for randomized smoothing, IRS. We show how to reuse the certification guarantees for the original smoothed model to certify an approximated model with very few samples. IRS significantly reduces the computational cost of certifying modified DNNs while maintaining strong robustness guarantees. We experimentally demonstrate the effectiveness of our approach, showing up to 3x certification speedup over the certification that applies randomized smoothing of the approximate model from scratch.



## **11. Indoor Location Fingerprinting Privacy: A Comprehensive Survey**

cs.CR

Submitted to ACM Computing Surveys

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.07345v1) [paper-pdf](http://arxiv.org/pdf/2404.07345v1)

**Authors**: Amir Fathalizadeh, Vahideh Moghtadaiee, Mina Alishahi

**Abstract**: The pervasive integration of Indoor Positioning Systems (IPS) arises from the limitations of Global Navigation Satellite Systems (GNSS) in indoor environments, leading to the widespread adoption of Location-Based Services (LBS). Specifically, indoor location fingerprinting employs diverse signal fingerprints from user devices, enabling precise location identification by Location Service Providers (LSP). Despite its broad applications across various domains, indoor location fingerprinting introduces a notable privacy risk, as both LSP and potential adversaries inherently have access to this sensitive information, compromising users' privacy. Consequently, concerns regarding privacy vulnerabilities in this context necessitate a focused exploration of privacy-preserving mechanisms. In response to these concerns, this survey presents a comprehensive review of Privacy-Preserving Mechanisms in Indoor Location Fingerprinting (ILFPPM) based on cryptographic, anonymization, differential privacy (DP), and federated learning (FL) techniques. We also propose a distinctive and novel grouping of privacy vulnerabilities, adversary and attack models, and available evaluation metrics specific to indoor location fingerprinting systems. Given the identified limitations and research gaps in this survey, we highlight numerous prospective opportunities for future investigation, aiming to motivate researchers interested in advancing this field. This survey serves as a valuable reference for researchers and provides a clear overview for those beyond this specific research domain.



## **12. Towards a Game-theoretic Understanding of Explanation-based Membership Inference Attacks**

cs.AI

arXiv admin note: text overlap with arXiv:2202.02659

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.07139v1) [paper-pdf](http://arxiv.org/pdf/2404.07139v1)

**Authors**: Kavita Kumari, Murtuza Jadliwala, Sumit Kumar Jha, Anindya Maiti

**Abstract**: Model explanations improve the transparency of black-box machine learning (ML) models and their decisions; however, they can also be exploited to carry out privacy threats such as membership inference attacks (MIA). Existing works have only analyzed MIA in a single "what if" interaction scenario between an adversary and the target ML model; thus, it does not discern the factors impacting the capabilities of an adversary in launching MIA in repeated interaction settings. Additionally, these works rely on assumptions about the adversary's knowledge of the target model's structure and, thus, do not guarantee the optimality of the predefined threshold required to distinguish the members from non-members. In this paper, we delve into the domain of explanation-based threshold attacks, where the adversary endeavors to carry out MIA attacks by leveraging the variance of explanations through iterative interactions with the system comprising of the target ML model and its corresponding explanation method. We model such interactions by employing a continuous-time stochastic signaling game framework. In our framework, an adversary plays a stopping game, interacting with the system (having imperfect information about the type of an adversary, i.e., honest or malicious) to obtain explanation variance information and computing an optimal threshold to determine the membership of a datapoint accurately. First, we propose a sound mathematical formulation to prove that such an optimal threshold exists, which can be used to launch MIA. Then, we characterize the conditions under which a unique Markov perfect equilibrium (or steady state) exists in this dynamic system. By means of a comprehensive set of simulations of the proposed game model, we assess different factors that can impact the capability of an adversary to launch MIA in such repeated interaction settings.



## **13. Adversarial purification for no-reference image-quality metrics: applicability study and new methods**

cs.CV

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06957v1) [paper-pdf](http://arxiv.org/pdf/2404.06957v1)

**Authors**: Aleksandr Gushchin, Anna Chistyakova, Vladislav Minashkin, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Recently, the area of adversarial attacks on image quality metrics has begun to be explored, whereas the area of defences remains under-researched. In this study, we aim to cover that case and check the transferability of adversarial purification defences from image classifiers to IQA methods. In this paper, we apply several widespread attacks on IQA models and examine the success of the defences against them. The purification methodologies covered different preprocessing techniques, including geometrical transformations, compression, denoising, and modern neural network-based methods. Also, we address the challenge of assessing the efficacy of a defensive methodology by proposing ways to estimate output visual quality and the success of neutralizing attacks. Defences were tested against attack on three IQA metrics -- Linearity, MetaIQA and SPAQ. The code for attacks and defences is available at: (link is hidden for a blind review).



## **14. Simpler becomes Harder: Do LLMs Exhibit a Coherent Behavior on Simplified Corpora?**

cs.CL

Published at DeTermIt! Workshop at LREC-COLING 2024

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06838v1) [paper-pdf](http://arxiv.org/pdf/2404.06838v1)

**Authors**: Miriam Anschütz, Edoardo Mosca, Georg Groh

**Abstract**: Text simplification seeks to improve readability while retaining the original content and meaning. Our study investigates whether pre-trained classifiers also maintain such coherence by comparing their predictions on both original and simplified inputs. We conduct experiments using 11 pre-trained models, including BERT and OpenAI's GPT 3.5, across six datasets spanning three languages. Additionally, we conduct a detailed analysis of the correlation between prediction change rates and simplification types/strengths. Our findings reveal alarming inconsistencies across all languages and models. If not promptly addressed, simplified inputs can be easily exploited to craft zero-iteration model-agnostic adversarial attacks with success rates of up to 50%



## **15. MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**

cs.LG

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2402.02263v2) [paper-pdf](http://arxiv.org/pdf/2402.02263v2)

**Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi

**Abstract**: Adversarial robustness often comes at the cost of degraded accuracy, impeding the real-life application of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.



## **16. Logit Calibration and Feature Contrast for Robust Federated Learning on Non-IID Data**

cs.LG

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06776v1) [paper-pdf](http://arxiv.org/pdf/2404.06776v1)

**Authors**: Yu Qiao, Chaoning Zhang, Apurba Adhikary, Choong Seon Hong

**Abstract**: Federated learning (FL) is a privacy-preserving distributed framework for collaborative model training on devices in edge networks. However, challenges arise due to vulnerability to adversarial examples (AEs) and the non-independent and identically distributed (non-IID) nature of data distribution among devices, hindering the deployment of adversarially robust and accurate learning models at the edge. While adversarial training (AT) is commonly acknowledged as an effective defense strategy against adversarial attacks in centralized training, we shed light on the adverse effects of directly applying AT in FL that can severely compromise accuracy, especially in non-IID challenges. Given this limitation, this paper proposes FatCC, which incorporates local logit \underline{C}alibration and global feature \underline{C}ontrast into the vanilla federated adversarial training (\underline{FAT}) process from both logit and feature perspectives. This approach can effectively enhance the federated system's robust accuracy (RA) and clean accuracy (CA). First, we propose logit calibration, where the logits are calibrated during local adversarial updates, thereby improving adversarial robustness. Second, FatCC introduces feature contrast, which involves a global alignment term that aligns each local representation with unbiased global features, thus further enhancing robustness and accuracy in federated adversarial environments. Extensive experiments across multiple datasets demonstrate that FatCC achieves comparable or superior performance gains in both CA and RA compared to other baselines.



## **17. False Claims against Model Ownership Resolution**

cs.CR

13pages,3 figures. To appear in the 33rd USENIX Security Symposium  (USENIX Security '24)

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2304.06607v7) [paper-pdf](http://arxiv.org/pdf/2304.06607v7)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation, we show that our false claim attacks always succeed in the MOR schemes that follow our generalization, including in a real-world model: Amazon's Rekognition API.



## **18. Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs**

cs.CR

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.07242v1) [paper-pdf](http://arxiv.org/pdf/2404.07242v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan

**Abstract**: Large Language Models (LLMs) are increasingly being developed and applied, but their widespread use faces challenges. These include aligning LLMs' responses with human values to prevent harmful outputs, which is addressed through safety training methods. Even so, bad actors and malicious users have succeeded in attempts to manipulate the LLMs to generate misaligned responses for harmful questions such as methods to create a bomb in school labs, recipes for harmful drugs, and ways to evade privacy rights. Another challenge is the multilingual capabilities of LLMs, which enable the model to understand and respond in multiple languages. Consequently, attackers exploit the unbalanced pre-training datasets of LLMs in different languages and the comparatively lower model performance in low-resource languages than high-resource ones. As a result, attackers use a low-resource languages to intentionally manipulate the model to create harmful responses. Many of the similar attack vectors have been patched by model providers, making the LLMs more robust against language-based manipulation. In this paper, we introduce a new black-box attack vector called the \emph{Sandwich attack}: a multi-language mixture attack, which manipulates state-of-the-art LLMs into generating harmful and misaligned responses. Our experiments with five different models, namely Google's Bard, Gemini Pro, LLaMA-2-70-B-Chat, GPT-3.5-Turbo, GPT-4, and Claude-3-OPUS, show that this attack vector can be used by adversaries to generate harmful responses and elicit misaligned responses from these models. By detailing both the mechanism and impact of the Sandwich attack, this paper aims to guide future research and development towards more secure and resilient LLMs, ensuring they serve the public good while minimizing potential for misuse.



## **19. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.16432v3) [paper-pdf](http://arxiv.org/pdf/2403.16432v3)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo. The resource is available at $\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.



## **20. LRR: Language-Driven Resamplable Continuous Representation against Adversarial Tracking Attacks**

cs.CV

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06247v1) [paper-pdf](http://arxiv.org/pdf/2404.06247v1)

**Authors**: Jianlang Chen, Xuhong Ren, Qing Guo, Felix Juefei-Xu, Di Lin, Wei Feng, Lei Ma, Jianjun Zhao

**Abstract**: Visual object tracking plays a critical role in visual-based autonomous systems, as it aims to estimate the position and size of the object of interest within a live video. Despite significant progress made in this field, state-of-the-art (SOTA) trackers often fail when faced with adversarial perturbations in the incoming frames. This can lead to significant robustness and security issues when these trackers are deployed in the real world. To achieve high accuracy on both clean and adversarial data, we propose building a spatial-temporal continuous representation using the semantic text guidance of the object of interest. This novel continuous representation enables us to reconstruct incoming frames to maintain semantic and appearance consistency with the object of interest and its clean counterparts. As a result, our proposed method successfully defends against different SOTA adversarial tracking attacks while maintaining high accuracy on clean data. In particular, our method significantly increases tracking accuracy under adversarial attacks with around 90% relative improvement on UAV123, which is even higher than the accuracy on clean data.



## **21. Towards Robust Domain Generation Algorithm Classification**

cs.CR

Accepted at ACM Asia Conference on Computer and Communications  Security (ASIA CCS 2024)

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06236v1) [paper-pdf](http://arxiv.org/pdf/2404.06236v1)

**Authors**: Arthur Drichel, Marc Meyer, Ulrike Meyer

**Abstract**: In this work, we conduct a comprehensive study on the robustness of domain generation algorithm (DGA) classifiers. We implement 32 white-box attacks, 19 of which are very effective and induce a false-negative rate (FNR) of $\approx$ 100\% on unhardened classifiers. To defend the classifiers, we evaluate different hardening approaches and propose a novel training scheme that leverages adversarial latent space vectors and discretized adversarial domains to significantly improve robustness. In our study, we highlight a pitfall to avoid when hardening classifiers and uncover training biases that can be easily exploited by attackers to bypass detection, but which can be mitigated by adversarial training (AT). In our study, we do not observe any trade-off between robustness and performance, on the contrary, hardening improves a classifier's detection performance for known and unknown DGAs. We implement all attacks and defenses discussed in this paper as a standalone library, which we make publicly available to facilitate hardening of DGA classifiers: https://gitlab.com/rwth-itsec/robust-dga-detection



## **22. FLEX: FLEXible Federated Learning Framework**

cs.CR

Submitted to Information Fusion

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06127v1) [paper-pdf](http://arxiv.org/pdf/2404.06127v1)

**Authors**: Francisco Herrera, Daniel Jiménez-López, Alberto Argente-Garrido, Nuria Rodríguez-Barroso, Cristina Zuheros, Ignacio Aguilera-Martos, Beatriz Bello, Mario García-Márquez, M. Victoria Luzón

**Abstract**: In the realm of Artificial Intelligence (AI), the need for privacy and security in data processing has become paramount. As AI applications continue to expand, the collection and handling of sensitive data raise concerns about individual privacy protection. Federated Learning (FL) emerges as a promising solution to address these challenges by enabling decentralized model training on local devices, thus preserving data privacy. This paper introduces FLEX: a FLEXible Federated Learning Framework designed to provide maximum flexibility in FL research experiments. By offering customizable features for data distribution, privacy parameters, and communication strategies, FLEX empowers researchers to innovate and develop novel FL techniques. The framework also includes libraries for specific FL implementations including: (1) anomalies, (2) blockchain, (3) adversarial attacks and defences, (4) natural language processing and (5) decision trees, enhancing its versatility and applicability in various domains. Overall, FLEX represents a significant advancement in FL research, facilitating the development of robust and efficient FL applications.



## **23. PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor**

cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.06668v2) [paper-pdf](http://arxiv.org/pdf/2403.06668v2)

**Authors**: Jaewon Jung, Hongsun Jang, Jaeyong Song, Jinho Lee

**Abstract**: Adversarial robustness of the neural network is a significant concern when it is applied to security-critical domains. In this situation, adversarial distillation is a promising option which aims to distill the robustness of the teacher network to improve the robustness of a small student network. Previous works pretrain the teacher network to make it robust to the adversarial examples aimed at itself. However, the adversarial examples are dependent on the parameters of the target network. The fixed teacher network inevitably degrades its robustness against the unseen transferred adversarial examples which targets the parameters of the student network in the adversarial distillation process. We propose PeerAiD to make a peer network learn the adversarial examples of the student network instead of adversarial examples aimed at itself. PeerAiD is an adversarial distillation that trains the peer network and the student network simultaneously in order to make the peer network specialized for defending the student network. We observe that such peer networks surpass the robustness of pretrained robust teacher network against student-attacked adversarial samples. With this peer network and adversarial distillation, PeerAiD achieves significantly higher robustness of the student network with AutoAttack (AA) accuracy up to 1.66%p and improves the natural accuracy of the student network up to 4.72%p with ResNet-18 and TinyImageNet dataset.



## **24. Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs**

cs.CV

Initial preprint. Under review

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06025v1) [paper-pdf](http://arxiv.org/pdf/2404.06025v1)

**Authors**: Zander W. Blasingame, Chen Liu

**Abstract**: Morphing attacks are an emerging threat to state-of-the-art Face Recognition (FR) systems, which aim to create a single image that contains the biometric information of multiple identities. Diffusion Morphs (DiM) are a recently proposed morphing attack that has achieved state-of-the-art performance for representation-based morphing attacks. However, none of the existing research on DiMs have leveraged the iterative nature of DiMs and left the DiM model as a black box, treating it no differently than one would a Generative Adversarial Network (GAN) or Varational AutoEncoder (VAE). We propose a greedy strategy on the iterative sampling process of DiM models which searches for an optimal step guided by an identity-based heuristic function. We compare our proposed algorithm against ten other state-of-the-art morphing algorithms using the open-source SYN-MAD 2022 competition dataset. We find that our proposed algorithm is unreasonably effective, fooling all of the tested FR systems with an MMPMR of 100%, outperforming all other morphing algorithms compared.



## **25. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

cs.LG

7 pages, 5 figures

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2307.03305v3) [paper-pdf](http://arxiv.org/pdf/2307.03305v3)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.



## **26. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

cs.LG

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2301.12554v4) [paper-pdf](http://arxiv.org/pdf/2301.12554v4)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that build neural classifiers robust against adversarial robustness, practitioners are still reluctant to adopt them due to their unacceptably severe clean accuracy penalties. This paper significantly alleviates this accuracy-robustness trade-off by mixing the output probabilities of a standard classifier and a robust classifier, where the standard network is optimized for clean accuracy and is not robust in general. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key to this improvement. In addition to providing intuitions and empirical evidence, we theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon = 8/255$) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.



## **27. Quantum Adversarial Learning for Kernel Methods**

quant-ph

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05824v1) [paper-pdf](http://arxiv.org/pdf/2404.05824v1)

**Authors**: Giuseppe Montalbano, Leonardo Banchi

**Abstract**: We show that hybrid quantum classifiers based on quantum kernel methods and support vector machines are vulnerable against adversarial attacks, namely small engineered perturbations of the input data can deceive the classifier into predicting the wrong result. Nonetheless, we also show that simple defence strategies based on data augmentation with a few crafted perturbations can make the classifier robust against new attacks. Our results find applications in security-critical learning problems and in mitigating the effect of some forms of quantum noise, since the attacker can also be understood as part of the surrounding environment.



## **28. Case Study: Neural Network Malware Detection Verification for Feature and Image Datasets**

cs.CR

In International Conference On Formal Methods in Software  Engineering, 2024; (FormaliSE'24)

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05703v1) [paper-pdf](http://arxiv.org/pdf/2404.05703v1)

**Authors**: Preston K. Robinette, Diego Manzanas Lopez, Serena Serbinowska, Kevin Leach, Taylor T. Johnson

**Abstract**: Malware, or software designed with harmful intent, is an ever-evolving threat that can have drastic effects on both individuals and institutions. Neural network malware classification systems are key tools for combating these threats but are vulnerable to adversarial machine learning attacks. These attacks perturb input data to cause misclassification, bypassing protective systems. Existing defenses often rely on enhancing the training process, thereby increasing the model's robustness to these perturbations, which is quantified using verification. While training improvements are necessary, we propose focusing on the verification process used to evaluate improvements to training. As such, we present a case study that evaluates a novel verification domain that will help to ensure tangible safeguards against adversaries and provide a more reliable means of evaluating the robustness and effectiveness of anti-malware systems. To do so, we describe malware classification and two types of common malware datasets (feature and image datasets), demonstrate the certified robustness accuracy of malware classifiers using the Neural Network Verification (NNV) and Neural Network Enumeration (nnenum) tools, and outline the challenges and future considerations necessary for the improvement and refinement of the verification of malware classification. By evaluating this novel domain as a case study, we hope to increase its visibility, encourage further research and scrutiny, and ultimately enhance the resilience of digital systems against malicious attacks.



## **29. David and Goliath: An Empirical Evaluation of Attacks and Defenses for QNNs at the Deep Edge**

cs.LG

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05688v1) [paper-pdf](http://arxiv.org/pdf/2404.05688v1)

**Authors**: Miguel Costa, Sandro Pinto

**Abstract**: ML is shifting from the cloud to the edge. Edge computing reduces the surface exposing private data and enables reliable throughput guarantees in real-time applications. Of the panoply of devices deployed at the edge, resource-constrained MCUs, e.g., Arm Cortex-M, are more prevalent, orders of magnitude cheaper, and less power-hungry than application processors or GPUs. Thus, enabling intelligence at the deep edge is the zeitgeist, with researchers focusing on unveiling novel approaches to deploy ANNs on these constrained devices. Quantization is a well-established technique that has proved effective in enabling the deployment of neural networks on MCUs; however, it is still an open question to understand the robustness of QNNs in the face of adversarial examples.   To fill this gap, we empirically evaluate the effectiveness of attacks and defenses from (full-precision) ANNs on (constrained) QNNs. Our evaluation includes three QNNs targeting TinyML applications, ten attacks, and six defenses. With this study, we draw a set of interesting findings. First, quantization increases the point distance to the decision boundary and leads the gradient estimated by some attacks to explode or vanish. Second, quantization can act as a noise attenuator or amplifier, depending on the noise magnitude, and causes gradient misalignment. Regarding adversarial defenses, we conclude that input pre-processing defenses show impressive results on small perturbations; however, they fall short as the perturbation increases. At the same time, train-based defenses increase the average point distance to the decision boundary, which holds after quantization. However, we argue that train-based defenses still need to smooth the quantization-shift and gradient misalignment phenomenons to counteract adversarial example transferability to QNNs. All artifacts are open-sourced to enable independent validation of results.



## **30. Investigating the Impact of Quantization on Adversarial Robustness**

cs.LG

Accepted to ICLR 2024 Workshop PML4LRS

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05639v1) [paper-pdf](http://arxiv.org/pdf/2404.05639v1)

**Authors**: Qun Li, Yuan Meng, Chen Tang, Jiacheng Jiang, Zhi Wang

**Abstract**: Quantization is a promising technique for reducing the bit-width of deep models to improve their runtime performance and storage efficiency, and thus becomes a fundamental step for deployment. In real-world scenarios, quantized models are often faced with adversarial attacks which cause the model to make incorrect inferences by introducing slight perturbations. However, recent studies have paid less attention to the impact of quantization on the model robustness. More surprisingly, existing studies on this topic even present inconsistent conclusions, which prompted our in-depth investigation. In this paper, we conduct a first-time analysis of the impact of the quantization pipeline components that can incorporate robust optimization under the settings of Post-Training Quantization and Quantization-Aware Training. Through our detailed analysis, we discovered that this inconsistency arises from the use of different pipelines in different studies, specifically regarding whether robust optimization is performed and at which quantization stage it occurs. Our research findings contribute insights into deploying more secure and robust quantized networks, assisting practitioners in reference for scenarios with high-security requirements and limited resources.



## **31. SoK: Gradient Leakage in Federated Learning**

cs.CR

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05403v1) [paper-pdf](http://arxiv.org/pdf/2404.05403v1)

**Authors**: Jiacheng Du, Jiahui Hu, Zhibo Wang, Peng Sun, Neil Zhenqiang Gong, Kui Ren

**Abstract**: Federated learning (FL) enables collaborative model training among multiple clients without raw data exposure. However, recent studies have shown that clients' private training data can be reconstructed from the gradients they share in FL, known as gradient inversion attacks (GIAs). While GIAs have demonstrated effectiveness under \emph{ideal settings and auxiliary assumptions}, their actual efficacy against \emph{practical FL systems} remains under-explored. To address this gap, we conduct a comprehensive study on GIAs in this work. We start with a survey of GIAs that establishes a milestone to trace their evolution and develops a systematization to uncover their inherent threats. Specifically, we categorize the auxiliary assumptions used by existing GIAs based on their practical accessibility to potential adversaries. To facilitate deeper analysis, we highlight the challenges that GIAs face in practical FL systems from three perspectives: \textit{local training}, \textit{model}, and \textit{post-processing}. We then perform extensive theoretical and empirical evaluations of state-of-the-art GIAs across diverse settings, utilizing eight datasets and thirteen models. Our findings indicate that GIAs have inherent limitations when reconstructing data under practical local training settings. Furthermore, their efficacy is sensitive to the trained model, and even simple post-processing measures applied to gradients can be effective defenses. Overall, our work provides crucial insights into the limited effectiveness of GIAs in practical FL systems. By rectifying prior misconceptions, we hope to inspire more accurate and realistic investigations on this topic.



## **32. BruSLeAttack: A Query-Efficient Score-Based Black-Box Sparse Adversarial Attack**

cs.LG

Published as a conference paper at the International Conference on  Learning Representations (ICLR 2024). Code is available at  https://brusliattack.github.io/

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05311v1) [paper-pdf](http://arxiv.org/pdf/2404.05311v1)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: We study the unique, less-well understood problem of generating sparse adversarial samples simply by observing the score-based replies to model queries. Sparse attacks aim to discover a minimum number-the l0 bounded-perturbations to model inputs to craft adversarial examples and misguide model decisions. But, in contrast to query-based dense attack counterparts against black-box models, constructing sparse adversarial perturbations, even when models serve confidence score information to queries in a score-based setting, is non-trivial. Because, such an attack leads to i) an NP-hard problem; and ii) a non-differentiable search space. We develop the BruSLeAttack-a new, faster (more query-efficient) Bayesian algorithm for the problem. We conduct extensive attack evaluations including an attack demonstration against a Machine Learning as a Service (MLaaS) offering exemplified by Google Cloud Vision and robustness testing of adversarial training regimes and a recent defense against black-box attacks. The proposed attack scales to achieve state-of-the-art attack success rates and query efficiency on standard computer vision tasks such as ImageNet across different model architectures. Our artefacts and DIY attack samples are available on GitHub. Importantly, our work facilitates faster evaluation of model vulnerabilities and raises our vigilance on the safety, security and reliability of deployed systems.



## **33. Out-of-Distribution Data: An Acquaintance of Adversarial Examples -- A Survey**

cs.LG

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05219v1) [paper-pdf](http://arxiv.org/pdf/2404.05219v1)

**Authors**: Naveen Karunanayake, Ravin Gunawardena, Suranga Seneviratne, Sanjay Chawla

**Abstract**: Deep neural networks (DNNs) deployed in real-world applications can encounter out-of-distribution (OOD) data and adversarial examples. These represent distinct forms of distributional shifts that can significantly impact DNNs' reliability and robustness. Traditionally, research has addressed OOD detection and adversarial robustness as separate challenges. This survey focuses on the intersection of these two areas, examining how the research community has investigated them together. Consequently, we identify two key research directions: robust OOD detection and unified robustness. Robust OOD detection aims to differentiate between in-distribution (ID) data and OOD data, even when they are adversarially manipulated to deceive the OOD detector. Unified robustness seeks a single approach to make DNNs robust against both adversarial attacks and OOD inputs. Accordingly, first, we establish a taxonomy based on the concept of distributional shifts. This framework clarifies how robust OOD detection and unified robustness relate to other research areas addressing distributional shifts, such as OOD detection, open set recognition, and anomaly detection. Subsequently, we review existing work on robust OOD detection and unified robustness. Finally, we highlight the limitations of the existing work and propose promising research directions that explore adversarial and OOD inputs within a unified framework.



## **34. Semantic Stealth: Adversarial Text Attacks on NLP Using Several Methods**

cs.CL

This report pertains to the Capstone Project done by Group 2 of the  Fall batch of 2023 students at Praxis Tech School, Kolkata, India. The  reports consists of 28 pages and it includes 10 tables. This is the preprint  which will be submitted to IEEE CONIT 2024 for review

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05159v1) [paper-pdf](http://arxiv.org/pdf/2404.05159v1)

**Authors**: Roopkatha Dey, Aivy Debnath, Sayak Kumar Dutta, Kaustav Ghosh, Arijit Mitra, Arghya Roy Chowdhury, Jaydip Sen

**Abstract**: In various real-world applications such as machine translation, sentiment analysis, and question answering, a pivotal role is played by NLP models, facilitating efficient communication and decision-making processes in domains ranging from healthcare to finance. However, a significant challenge is posed to the robustness of these natural language processing models by text adversarial attacks. These attacks involve the deliberate manipulation of input text to mislead the predictions of the model while maintaining human interpretability. Despite the remarkable performance achieved by state-of-the-art models like BERT in various natural language processing tasks, they are found to remain vulnerable to adversarial perturbations in the input text. In addressing the vulnerability of text classifiers to adversarial attacks, three distinct attack mechanisms are explored in this paper using the victim model BERT: BERT-on-BERT attack, PWWS attack, and Fraud Bargain's Attack (FBA). Leveraging the IMDB, AG News, and SST2 datasets, a thorough comparative analysis is conducted to assess the effectiveness of these attacks on the BERT classifier model. It is revealed by the analysis that PWWS emerges as the most potent adversary, consistently outperforming other methods across multiple evaluation scenarios, thereby emphasizing its efficacy in generating adversarial examples for text classification. Through comprehensive experimentation, the performance of these attacks is assessed and the findings indicate that the PWWS attack outperforms others, demonstrating lower runtime, higher accuracy, and favorable semantic similarity scores. The key insight of this paper lies in the assessment of the relative performances of three prevalent state-of-the-art attack mechanisms.



## **35. Enabling Privacy-Preserving Cyber Threat Detection with Federated Learning**

cs.CR

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05130v1) [paper-pdf](http://arxiv.org/pdf/2404.05130v1)

**Authors**: Yu Bi, Yekai Li, Xuan Feng, Xianghang Mi

**Abstract**: Despite achieving good performance and wide adoption, machine learning based security detection models (e.g., malware classifiers) are subject to concept drift and evasive evolution of attackers, which renders up-to-date threat data as a necessity. However, due to enforcement of various privacy protection regulations (e.g., GDPR), it is becoming increasingly challenging or even prohibitive for security vendors to collect individual-relevant and privacy-sensitive threat datasets, e.g., SMS spam/non-spam messages from mobile devices. To address such obstacles, this study systematically profiles the (in)feasibility of federated learning for privacy-preserving cyber threat detection in terms of effectiveness, byzantine resilience, and efficiency. This is made possible by the build-up of multiple threat datasets and threat detection models, and more importantly, the design of realistic and security-specific experiments.   We evaluate FL on two representative threat detection tasks, namely SMS spam detection and Android malware detection. It shows that FL-trained detection models can achieve a performance that is comparable to centrally trained counterparts. Also, most non-IID data distributions have either minor or negligible impact on the model performance, while a label-based non-IID distribution of a high extent can incur non-negligible fluctuation and delay in FL training. Then, under a realistic threat model, FL turns out to be adversary-resistant to attacks of both data poisoning and model poisoning. Particularly, the attacking impact of a practical data poisoning attack is no more than 0.14\% loss in model accuracy. Regarding FL efficiency, a bootstrapping strategy turns out to be effective to mitigate the training delay as observed in label-based non-IID scenarios.



## **36. Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations**

cs.LG

29 pages, 4 figures

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2402.05713v3) [paper-pdf](http://arxiv.org/pdf/2402.05713v3)

**Authors**: Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh

**Abstract**: The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce undetectable underdiagnosis bias in DL models. Our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups show that adversarial bias attacks demonstrate high-selectivity for bias in the targeted group by degrading group model performance without impacting overall model performance. Furthermore, our results indicate that adversarial bias attacks result in biased DL models that propagate prediction bias even when evaluated with external datasets.



## **37. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2402.08656v3) [paper-pdf](http://arxiv.org/pdf/2402.08656v3)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.



## **38. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2311.08268v4) [paper-pdf](http://arxiv.org/pdf/2311.08268v4)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.



## **39. Provable Robustness Against a Union of $\ell_0$ Adversarial Attacks**

cs.LG

Accepted at AAAI 2024 -- Extended version including the supplementary  material

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2302.11628v4) [paper-pdf](http://arxiv.org/pdf/2302.11628v4)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Sparse or $\ell_0$ adversarial attacks arbitrarily perturb an unknown subset of the features. $\ell_0$ robustness analysis is particularly well-suited for heterogeneous (tabular) data where features have different types or scales. State-of-the-art $\ell_0$ certified defenses are based on randomized smoothing and apply to evasion attacks only. This paper proposes feature partition aggregation (FPA) -- a certified defense against the union of $\ell_0$ evasion, backdoor, and poisoning attacks. FPA generates its stronger robustness guarantees via an ensemble whose submodels are trained on disjoint feature sets. Compared to state-of-the-art $\ell_0$ defenses, FPA is up to 3,000${\times}$ faster and provides larger median robustness guarantees (e.g., median certificates of 13 pixels over 10 for CIFAR10, 12 pixels over 10 for MNIST, 4 features over 1 for Weather, and 3 features over 1 for Ames), meaning FPA provides the additional dimensions of robustness essentially for free.



## **40. Data Poisoning Attacks on Off-Policy Policy Evaluation Methods**

cs.LG

Accepted at UAI 2022

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04714v1) [paper-pdf](http://arxiv.org/pdf/2404.04714v1)

**Authors**: Elita Lobo, Harvineet Singh, Marek Petrik, Cynthia Rudin, Himabindu Lakkaraju

**Abstract**: Off-policy Evaluation (OPE) methods are a crucial tool for evaluating policies in high-stakes domains such as healthcare, where exploration is often infeasible, unethical, or expensive. However, the extent to which such methods can be trusted under adversarial threats to data quality is largely unexplored. In this work, we make the first attempt at investigating the sensitivity of OPE methods to marginal adversarial perturbations to the data. We design a generic data poisoning attack framework leveraging influence functions from robust statistics to carefully construct perturbations that maximize error in the policy value estimates. We carry out extensive experimentation with multiple healthcare and control datasets. Our results demonstrate that many existing OPE methods are highly prone to generating value estimates with large errors when subject to data poisoning attacks, even for small adversarial perturbations. These findings question the reliability of policy values derived using OPE methods and motivate the need for developing OPE methods that are statistically robust to train-time data poisoning attacks.



## **41. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

cs.CL

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2310.00322v4) [paper-pdf](http://arxiv.org/pdf/2310.00322v4)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.



## **42. CANEDERLI: On The Impact of Adversarial Training and Transferability on CAN Intrusion Detection Systems**

cs.CR

Accepted at WiseML 2024

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04648v1) [paper-pdf](http://arxiv.org/pdf/2404.04648v1)

**Authors**: Francesco Marchiori, Mauro Conti

**Abstract**: The growing integration of vehicles with external networks has led to a surge in attacks targeting their Controller Area Network (CAN) internal bus. As a countermeasure, various Intrusion Detection Systems (IDSs) have been suggested in the literature to prevent and mitigate these threats. With the increasing volume of data facilitated by the integration of Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication networks, most of these systems rely on data-driven approaches such as Machine Learning (ML) and Deep Learning (DL) models. However, these systems are susceptible to adversarial evasion attacks. While many researchers have explored this vulnerability, their studies often involve unrealistic assumptions, lack consideration for a realistic threat model, and fail to provide effective solutions.   In this paper, we present CANEDERLI (CAN Evasion Detection ResiLIence), a novel framework for securing CAN-based IDSs. Our system considers a realistic threat model and addresses the impact of adversarial attacks on DL-based detection systems. Our findings highlight strong transferability properties among diverse attack methodologies by considering multiple state-of-the-art attacks and model architectures. We analyze the impact of adversarial training in addressing this threat and propose an adaptive online adversarial training technique outclassing traditional fine-tuning methodologies with F1 scores up to 0.941. By making our framework publicly available, we aid practitioners and researchers in assessing the resilience of IDSs to a varied adversarial landscape.



## **43. Dynamic Graph Information Bottleneck**

cs.LG

Accepted by the research tracks of The Web Conference 2024 (WWW 2024)

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2402.06716v3) [paper-pdf](http://arxiv.org/pdf/2402.06716v3)

**Authors**: Haonan Yuan, Qingyun Sun, Xingcheng Fu, Cheng Ji, Jianxin Li

**Abstract**: Dynamic Graphs widely exist in the real world, which carry complicated spatial and temporal feature patterns, challenging their representation learning. Dynamic Graph Neural Networks (DGNNs) have shown impressive predictive abilities by exploiting the intrinsic dynamics. However, DGNNs exhibit limited robustness, prone to adversarial attacks. This paper presents the novel Dynamic Graph Information Bottleneck (DGIB) framework to learn robust and discriminative representations. Leveraged by the Information Bottleneck (IB) principle, we first propose the expected optimal representations should satisfy the Minimal-Sufficient-Consensual (MSC) Condition. To compress redundant as well as conserve meritorious information into latent representation, DGIB iteratively directs and refines the structural and feature information flow passing through graph snapshots. To meet the MSC Condition, we decompose the overall IB objectives into DGIB$_{MS}$ and DGIB$_C$, in which the DGIB$_{MS}$ channel aims to learn the minimal and sufficient representations, with the DGIB$_{MS}$ channel guarantees the predictive consensus. Extensive experiments on real-world and synthetic dynamic graph datasets demonstrate the superior robustness of DGIB against adversarial attacks compared with state-of-the-art baselines in the link prediction task. To the best of our knowledge, DGIB is the first work to learn robust representations of dynamic graphs grounded in the information-theoretic IB principle.



## **44. Goal-guided Generative Prompt Injection Attack on Large Language Models**

cs.CR

22 pages, 8 figures

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.07234v1) [paper-pdf](http://arxiv.org/pdf/2404.07234v1)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.



## **45. Recovery from Adversarial Attacks in Cyber-physical Systems: Shallow, Deep and Exploratory Works**

eess.SY

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04472v1) [paper-pdf](http://arxiv.org/pdf/2404.04472v1)

**Authors**: Pengyuan Lu, Lin Zhang, Mengyu Liu, Kaustubh Sridhar, Fanxin Kong, Oleg Sokolsky, Insup Lee

**Abstract**: Cyber-physical systems (CPS) have experienced rapid growth in recent decades. However, like any other computer-based systems, malicious attacks evolve mutually, driving CPS to undesirable physical states and potentially causing catastrophes. Although the current state-of-the-art is well aware of this issue, the majority of researchers have not focused on CPS recovery, the procedure we defined as restoring a CPS's physical state back to a target condition under adversarial attacks. To call for attention on CPS recovery and identify existing efforts, we have surveyed a total of 30 relevant papers. We identify a major partition of the proposed recovery strategies: shallow recovery vs. deep recovery, where the former does not use a dedicated recovery controller while the latter does. Additionally, we surveyed exploratory research on topics that facilitate recovery. From these publications, we discuss the current state-of-the-art of CPS recovery, with respect to applications, attack type, attack surfaces and system dynamics. Then, we identify untouched sub-domains in this field and suggest possible future directions for researchers.



## **46. Increased LLM Vulnerabilities from Fine-tuning and Quantization**

cs.CR

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04392v1) [paper-pdf](http://arxiv.org/pdf/2404.04392v1)

**Authors**: Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Large Language Models (LLMs) have become very popular and have found use cases in many domains, such as chatbots, auto-task completion agents, and much more. However, LLMs are vulnerable to different types of attacks, such as jailbreaking, prompt injection attacks, and privacy leakage attacks. Foundational LLMs undergo adversarial and alignment training to learn not to generate malicious and toxic content. For specialized use cases, these foundational LLMs are subjected to fine-tuning or quantization for better performance and efficiency. We examine the impact of downstream tasks such as fine-tuning and quantization on LLM vulnerability. We test foundation models like Mistral, Llama, MosaicML, and their fine-tuned versions. Our research shows that fine-tuning and quantization reduces jailbreak resistance significantly, leading to increased LLM vulnerabilities. Finally, we demonstrate the utility of external guardrails in reducing LLM vulnerabilities.



## **47. Compositional Estimation of Lipschitz Constants for Deep Neural Networks**

cs.LG

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04375v1) [paper-pdf](http://arxiv.org/pdf/2404.04375v1)

**Authors**: Yuezhu Xu, S. Sivaranjani

**Abstract**: The Lipschitz constant plays a crucial role in certifying the robustness of neural networks to input perturbations and adversarial attacks, as well as the stability and safety of systems with neural network controllers. Therefore, estimation of tight bounds on the Lipschitz constant of neural networks is a well-studied topic. However, typical approaches involve solving a large matrix verification problem, the computational cost of which grows significantly for deeper networks. In this letter, we provide a compositional approach to estimate Lipschitz constants for deep feedforward neural networks by obtaining an exact decomposition of the large matrix verification problem into smaller sub-problems. We further obtain a closed-form solution that applies to most common neural network activation functions, which will enable rapid robustness and stability certificates for neural networks deployed in online control settings. Finally, we demonstrate through numerical experiments that our approach provides a steep reduction in computation time while yielding Lipschitz bounds that are very close to those achieved by state-of-the-art approaches.



## **48. Dissecting Distribution Inference**

cs.LG

Accepted at SaTML 2023 (updated Yifu's email address)

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2212.07591v2) [paper-pdf](http://arxiv.org/pdf/2212.07591v2)

**Authors**: Anshuman Suri, Yifu Lu, Yanjin Chen, David Evans

**Abstract**: A distribution inference attack aims to infer statistical properties of data used to train machine learning models. These attacks are sometimes surprisingly potent, but the factors that impact distribution inference risk are not well understood and demonstrated attacks often rely on strong and unrealistic assumptions such as full knowledge of training environments even in supposedly black-box threat scenarios. To improve understanding of distribution inference risks, we develop a new black-box attack that even outperforms the best known white-box attack in most settings. Using this new attack, we evaluate distribution inference risk while relaxing a variety of assumptions about the adversary's knowledge under black-box access, like known model architectures and label-only access. Finally, we evaluate the effectiveness of previously proposed defenses and introduce new defenses. We find that although noise-based defenses appear to be ineffective, a simple re-sampling defense can be highly effective. Code is available at https://github.com/iamgroot42/dissecting_distribution_inference



## **49. Evaluating Adversarial Robustness: A Comparison Of FGSM, Carlini-Wagner Attacks, And The Role of Distillation as Defense Mechanism**

cs.CR

This report pertains to the Capstone Project done by Group 1 of the  Fall batch of 2023 students at Praxis Tech School, Kolkata, India. The  reports consists of 35 pages and it includes 15 figures and 10 tables. This  is the preprint which will be submitted to to an IEEE international  conference for review

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04245v1) [paper-pdf](http://arxiv.org/pdf/2404.04245v1)

**Authors**: Trilokesh Ranjan Sarkar, Nilanjan Das, Pralay Sankar Maitra, Bijoy Some, Ritwik Saha, Orijita Adhikary, Bishal Bose, Jaydip Sen

**Abstract**: This technical report delves into an in-depth exploration of adversarial attacks specifically targeted at Deep Neural Networks (DNNs) utilized for image classification. The study also investigates defense mechanisms aimed at bolstering the robustness of machine learning models. The research focuses on comprehending the ramifications of two prominent attack methodologies: the Fast Gradient Sign Method (FGSM) and the Carlini-Wagner (CW) approach. These attacks are examined concerning three pre-trained image classifiers: Resnext50_32x4d, DenseNet-201, and VGG-19, utilizing the Tiny-ImageNet dataset. Furthermore, the study proposes the robustness of defensive distillation as a defense mechanism to counter FGSM and CW attacks. This defense mechanism is evaluated using the CIFAR-10 dataset, where CNN models, specifically resnet101 and Resnext50_32x4d, serve as the teacher and student models, respectively. The proposed defensive distillation model exhibits effectiveness in thwarting attacks such as FGSM. However, it is noted to remain susceptible to more sophisticated techniques like the CW attack. The document presents a meticulous validation of the proposed scheme. It provides detailed and comprehensive results, elucidating the efficacy and limitations of the defense mechanisms employed. Through rigorous experimentation and analysis, the study offers insights into the dynamics of adversarial attacks on DNNs, as well as the effectiveness of defensive strategies in mitigating their impact.



## **50. On Inherent Adversarial Robustness of Active Vision Systems**

cs.CV

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.00185v2) [paper-pdf](http://arxiv.org/pdf/2404.00185v2)

**Authors**: Amitangshu Mukherjee, Timur Ibrayev, Kaushik Roy

**Abstract**: Current Deep Neural Networks are vulnerable to adversarial examples, which alter their predictions by adding carefully crafted noise. Since human eyes are robust to such inputs, it is possible that the vulnerability stems from the standard way of processing inputs in one shot by processing every pixel with the same importance. In contrast, neuroscience suggests that the human vision system can differentiate salient features by (1) switching between multiple fixation points (saccades) and (2) processing the surrounding with a non-uniform external resolution (foveation). In this work, we advocate that the integration of such active vision mechanisms into current deep learning systems can offer robustness benefits. Specifically, we empirically demonstrate the inherent robustness of two active vision methods - GFNet and FALcon - under a black box threat model. By learning and inferencing based on downsampled glimpses obtained from multiple distinct fixation points within an input, we show that these active methods achieve (2-3) times greater robustness compared to a standard passive convolutional network under state-of-the-art adversarial attacks. More importantly, we provide illustrative and interpretable visualization analysis that demonstrates how performing inference from distinct fixation points makes active vision methods less vulnerable to malicious inputs.



