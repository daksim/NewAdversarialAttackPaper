# Latest Adversarial Attack Papers
**update at 2025-01-29 20:07:08**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Scanning Trojaned Models Using Out-of-Distribution Samples**

cs.LG

Accepted at the Thirty-Eighth Annual Conference on Neural Information  Processing Systems (NeurIPS) 2024. The code repository is available at:  https://github.com/rohban-lab/TRODO

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.17151v1) [paper-pdf](http://arxiv.org/pdf/2501.17151v1)

**Authors**: Hossein Mirzaei, Ali Ansari, Bahar Dibaei Nia, Mojtaba Nafez, Moein Madadi, Sepehr Rezaee, Zeinab Sadat Taghavi, Arad Maleki, Kian Shamsaie, Mahdi Hajialilue, Jafar Habibi, Mohammad Sabokrou, Mohammad Hossein Rohban

**Abstract**: Scanning for trojan (backdoor) in deep neural networks is crucial due to their significant real-world applications. There has been an increasing focus on developing effective general trojan scanning methods across various trojan attacks. Despite advancements, there remains a shortage of methods that perform effectively without preconceived assumptions about the backdoor attack method. Additionally, we have observed that current methods struggle to identify classifiers trojaned using adversarial training. Motivated by these challenges, our study introduces a novel scanning method named TRODO (TROjan scanning by Detection of adversarial shifts in Out-of-distribution samples). TRODO leverages the concept of "blind spots"--regions where trojaned classifiers erroneously identify out-of-distribution (OOD) samples as in-distribution (ID). We scan for these blind spots by adversarially shifting OOD samples towards in-distribution. The increased likelihood of perturbed OOD samples being classified as ID serves as a signature for trojan detection. TRODO is both trojan and label mapping agnostic, effective even against adversarially trained trojaned classifiers. It is applicable even in scenarios where training data is absent, demonstrating high accuracy and adaptability across various scenarios and datasets, highlighting its potential as a robust trojan scanning strategy.



## **2. Hybrid Deep Learning Model for Multiple Cache Side Channel Attacks Detection: A Comparative Analysis**

cs.CR

8 pages, 4 figures. Accepted in IEEE's 2nd International Conference  on Computational Intelligence and Network Systems

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.17123v1) [paper-pdf](http://arxiv.org/pdf/2501.17123v1)

**Authors**: Tejal Joshi, Aarya Kawalay, Anvi Jamkhande, Amit Joshi

**Abstract**: Cache side channel attacks are a sophisticated and persistent threat that exploit vulnerabilities in modern processors to extract sensitive information. These attacks leverage weaknesses in shared computational resources, particularly the last level cache, to infer patterns in data access and execution flows, often bypassing traditional security defenses. Such attacks are especially dangerous as they can be executed remotely without requiring physical access to the victim's device. This study focuses on a specific class of these threats: fingerprinting attacks, where an adversary monitors and analyzes the behavior of co-located processes via cache side channels. This can potentially reveal confidential information, such as encryption keys or user activity patterns. A comprehensive threat model illustrates how attackers sharing computational resources with target systems exploit these side channels to compromise sensitive data. To mitigate such risks, a hybrid deep learning model is proposed for detecting cache side channel attacks. Its performance is compared with five widely used deep learning models: Multi-Layer Perceptron, Convolutional Neural Network, Simple Recurrent Neural Network, Long Short-Term Memory, and Gated Recurrent Unit. The experimental results demonstrate that the hybrid model achieves a detection rate of up to 99.96%. These findings highlight the limitations of existing models, the need for enhanced defensive mechanisms, and directions for future research to secure sensitive data against evolving side channel threats.



## **3. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

AISTATS 2025

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2412.08099v3) [paper-pdf](http://arxiv.org/pdf/2412.08099v3)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.



## **4. Random-Set Neural Networks (RS-NN)**

cs.LG

Published as a conference paper at the Thirteenth International  Conference on Learning Representations (ICLR 2025)

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2307.05772v4) [paper-pdf](http://arxiv.org/pdf/2307.05772v4)

**Authors**: Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, Fabio Cuzzolin

**Abstract**: Machine learning is increasingly deployed in safety-critical domains where erroneous predictions may lead to potentially catastrophic consequences, highlighting the need for learning systems to be aware of how confident they are in their own predictions: in other words, 'to know when they do not know'. In this paper, we propose a novel Random-Set Neural Network (RS-NN) approach to classification which predicts belief functions (rather than classical probability vectors) over the class list using the mathematics of random sets, i.e., distributions over the collection of sets of classes. RS-NN encodes the 'epistemic' uncertainty induced by training sets that are insufficiently representative or limited in size via the size of the convex set of probability vectors associated with a predicted belief function. Our approach outperforms state-of-the-art Bayesian and Ensemble methods in terms of accuracy, uncertainty estimation and out-of-distribution (OoD) detection on multiple benchmarks (CIFAR-10 vs SVHN/Intel-Image, MNIST vs FMNIST/KMNIST, ImageNet vs ImageNet-O). RS-NN also scales up effectively to large-scale architectures (e.g. WideResNet-28-10, VGG16, Inception V3, EfficientNetB2 and ViT-Base-16), exhibits remarkable robustness to adversarial attacks and can provide statistical guarantees in a conformal learning setting.



## **5. Adversarial Masked Autoencoder Purifier with Defense Transferability**

cs.CV

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16904v1) [paper-pdf](http://arxiv.org/pdf/2501.16904v1)

**Authors**: Yuan-Chih Chen, Chun-Shien Lu

**Abstract**: The study of adversarial defense still struggles to combat with advanced adversarial attacks. In contrast to most prior studies that rely on the diffusion model for test-time defense to remarkably increase the inference time, we propose Masked AutoEncoder Purifier (MAEP), which integrates Masked AutoEncoder (MAE) into an adversarial purifier framework for test-time purification. While MAEP achieves promising adversarial robustness, it particularly features model defense transferability and attack generalization without relying on using additional data that is different from the training dataset. To our knowledge, MAEP is the first study of adversarial purifier based on MAE. Extensive experimental results demonstrate that our method can not only maintain clear accuracy with only a slight drop but also exhibit a close gap between the clean and robust accuracy. Notably, MAEP trained on CIFAR10 achieves state-of-the-art performance even when tested directly on ImageNet, outperforming existing diffusion-based models trained specifically on ImageNet.



## **6. Document Screenshot Retrievers are Vulnerable to Pixel Poisoning Attacks**

cs.IR

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16902v1) [paper-pdf](http://arxiv.org/pdf/2501.16902v1)

**Authors**: Shengyao Zhuang, Ekaterina Khramtsova, Xueguang Ma, Bevan Koopman, Jimmy Lin, Guido Zuccon

**Abstract**: Recent advancements in dense retrieval have introduced vision-language model (VLM)-based retrievers, such as DSE and ColPali, which leverage document screenshots embedded as vectors to enable effective search and offer a simplified pipeline over traditional text-only methods. In this study, we propose three pixel poisoning attack methods designed to compromise VLM-based retrievers and evaluate their effectiveness under various attack settings and parameter configurations. Our empirical results demonstrate that injecting even a single adversarial screenshot into the retrieval corpus can significantly disrupt search results, poisoning the top-10 retrieved documents for 41.9% of queries in the case of DSE and 26.4% for ColPali. These vulnerability rates notably exceed those observed with equivalent attacks on text-only retrievers. Moreover, when targeting a small set of known queries, the attack success rate raises, achieving complete success in certain cases. By exposing the vulnerabilities inherent in vision-language models, this work highlights the potential risks associated with their deployment.



## **7. "My Whereabouts, my Location, it's Directly Linked to my Physical Security": An Exploratory Qualitative Study of Location-Dependent Security and Privacy Perceptions among Activist Tech Users**

cs.HC

10 pages, incl. interview guide and codebook

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16885v1) [paper-pdf](http://arxiv.org/pdf/2501.16885v1)

**Authors**: Christian Eichenmüller, Lisa Kuhn, Zinaida Benenson

**Abstract**: Digital-safety research with at-risk users is particularly urgent. At-risk users are more likely to be digitally attacked or targeted by surveillance and could be disproportionately harmed by attacks that facilitate physical assaults. One group of such at-risk users are activists and politically active individuals. For them, as for other at-risk users, the rise of smart environments harbors new risks. Since digitization and datafication are no longer limited to a series of personal devices that can be switched on and off, but increasingly and continuously surround users, granular geolocation poses new safety challenges. Drawing on eight exploratory qualitative interviews of an ongoing research project, this contribution highlights what activists with powerful adversaries think about evermore data traces, including location data, and how they intend to deal with emerging risks. Responses of activists include attempts to control one's immediate technological surroundings and to more carefully manage device-related location data. For some activists, threat modeling has also shaped provider choices based on geopolitical considerations. Since many activists have not enough digital-safety knowledge for effective protection, feelings of insecurity and paranoia are widespread. Channeling the concerns and fears of our interlocutors, we call for more research on how activists can protect themselves against evermore fine-grained location data tracking.



## **8. CantorNet: A Sandbox for Testing Geometrical and Topological Complexity Measures**

cs.NE

Accepted at the NeurIPS Workshop on Symmetry and Geometry in Neural  Representations, 2024

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2411.19713v3) [paper-pdf](http://arxiv.org/pdf/2411.19713v3)

**Authors**: Michal Lewandowski, Hamid Eghbalzadeh, Bernhard A. Moser

**Abstract**: Many natural phenomena are characterized by self-similarity, for example the symmetry of human faces, or a repetitive motif of a song. Studying of such symmetries will allow us to gain deeper insights into the underlying mechanisms of complex systems. Recognizing the importance of understanding these patterns, we propose a geometrically inspired framework to study such phenomena in artificial neural networks. To this end, we introduce \emph{CantorNet}, inspired by the triadic construction of the Cantor set, which was introduced by Georg Cantor in the $19^\text{th}$ century. In mathematics, the Cantor set is a set of points lying on a single line that is self-similar and has a counter intuitive property of being an uncountably infinite null set. Similarly, we introduce CantorNet as a sandbox for studying self-similarity by means of novel topological and geometrical complexity measures. CantorNet constitutes a family of ReLU neural networks that spans the whole spectrum of possible Kolmogorov complexities, including the two opposite descriptions (linear and exponential as measured by the description length). CantorNet's decision boundaries can be arbitrarily ragged, yet are analytically known. Besides serving as a testing ground for complexity measures, our work may serve to illustrate potential pitfalls in geometry-ignorant data augmentation techniques and adversarial attacks.



## **9. Bones of Contention: Exploring Query-Efficient Attacks Against Skeleton Recognition Systems**

cs.CR

13 pages, 13 figures

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16843v1) [paper-pdf](http://arxiv.org/pdf/2501.16843v1)

**Authors**: Yuxin Cao, Kai Ye, Derui Wang, Minhui Xue, Hao Ge, Chenxiong Qian, Jin Song Dong

**Abstract**: Skeleton action recognition models have secured more attention than video-based ones in various applications due to privacy preservation and lower storage requirements. Skeleton data are typically transmitted to cloud servers for action recognition, with results returned to clients via Apps/APIs. However, the vulnerability of skeletal models against adversarial perturbations gradually reveals the unreliability of these systems. Existing black-box attacks all operate in a decision-based manner, resulting in numerous queries that hinder efficiency and feasibility in real-world applications. Moreover, all attacks off the shelf focus on only restricted perturbations, while ignoring model weaknesses when encountered with non-semantic perturbations. In this paper, we propose two query-effIcient Skeletal Adversarial AttaCks, ISAAC-K and ISAAC-N. As a black-box attack, ISAAC-K utilizes Grad-CAM in a surrogate model to extract key joints where minor sparse perturbations are then added to fool the classifier. To guarantee natural adversarial motions, we introduce constraints of both bone length and temporal consistency. ISAAC-K finds stronger adversarial examples on $\ell_\infty$ norm, which can encompass those on other norms. Exhaustive experiments substantiate that ISAAC-K can uplift the attack efficiency of the perturbations under 10 skeletal models. Additionally, as a byproduct, ISAAC-N fools the classifier by replacing skeletons unrelated to the action. We surprisingly find that skeletal models are vulnerable to large perturbations where the part-wise non-semantic joints are just replaced, leading to a query-free no-box attack without any prior knowledge. Based on that, four adaptive defenses are eventually proposed to improve the robustness of skeleton recognition models.



## **10. HateBench: Benchmarking Hate Speech Detectors on LLM-Generated Content and Hate Campaigns**

cs.CR

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16750v1) [paper-pdf](http://arxiv.org/pdf/2501.16750v1)

**Authors**: Xinyue Shen, Yixin Wu, Yiting Qu, Michael Backes, Savvas Zannettou, Yang Zhang

**Abstract**: Large Language Models (LLMs) have raised increasing concerns about their misuse in generating hate speech. Among all the efforts to address this issue, hate speech detectors play a crucial role. However, the effectiveness of different detectors against LLM-generated hate speech remains largely unknown. In this paper, we propose HateBench, a framework for benchmarking hate speech detectors on LLM-generated hate speech. We first construct a hate speech dataset of 7,838 samples generated by six widely-used LLMs covering 34 identity groups, with meticulous annotations by three labelers. We then assess the effectiveness of eight representative hate speech detectors on the LLM-generated dataset. Our results show that while detectors are generally effective in identifying LLM-generated hate speech, their performance degrades with newer versions of LLMs. We also reveal the potential of LLM-driven hate campaigns, a new threat that LLMs bring to the field of hate speech detection. By leveraging advanced techniques like adversarial attacks and model stealing attacks, the adversary can intentionally evade the detector and automate hate campaigns online. The most potent adversarial attack achieves an attack success rate of 0.966, and its attack efficiency can be further improved by $13-21\times$ through model stealing attacks with acceptable attack performance. We hope our study can serve as a call to action for the research community and platform moderators to fortify defenses against these emerging threats.



## **11. Blockchain Address Poisoning**

cs.CR

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16681v1) [paper-pdf](http://arxiv.org/pdf/2501.16681v1)

**Authors**: Taro Tsuchiya, Jin-Dong Dong, Kyle Soska, Nicolas Christin

**Abstract**: In many blockchains, e.g., Ethereum, Binance Smart Chain (BSC), the primary representation used for wallet addresses is a hardly memorable 40-digit hexadecimal string. As a result, users often select addresses from their recent transaction history, which enables blockchain address poisoning. The adversary first generates lookalike addresses similar to one with which the victim has previously interacted, and then engages with the victim to ``poison'' their transaction history. The goal is to have the victim mistakenly send tokens to the lookalike address, as opposed to the intended recipient. Compared to contemporary studies, this paper provides four notable contributions. First, we develop a detection system and perform measurements over two years on Ethereum and BSC. We identify 13 times the number of attack attempts reported previously -- totaling 270M on-chain attacks targeting 17M victims. 6,633 incidents have caused at least 83.8M USD in losses, which makes blockchain address poisoning one of the largest cryptocurrency phishing schemes observed in the wild. Second, we analyze a few large attack entities using improved clustering techniques, and model attacker profitability and competition. Third, we reveal attack strategies -- targeted populations, success conditions (address similarity, timing), and cross-chain attacks. Fourth, we mathematically define and simulate the lookalike address-generation process across various software- and hardware-based implementations, and identify a large-scale attacker group that appears to use GPUs. We also discuss defensive countermeasures.



## **12. Data-Free Model-Related Attacks: Unleashing the Potential of Generative AI**

cs.CR

Accepted at USENIX Security 2025

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16671v1) [paper-pdf](http://arxiv.org/pdf/2501.16671v1)

**Authors**: Dayong Ye, Tianqing Zhu, Shang Wang, Bo Liu, Leo Yu Zhang, Wanlei Zhou, Yang Zhang

**Abstract**: Generative AI technology has become increasingly integrated into our daily lives, offering powerful capabilities to enhance productivity. However, these same capabilities can be exploited by adversaries for malicious purposes. While existing research on adversarial applications of generative AI predominantly focuses on cyberattacks, less attention has been given to attacks targeting deep learning models. In this paper, we introduce the use of generative AI for facilitating model-related attacks, including model extraction, membership inference, and model inversion. Our study reveals that adversaries can launch a variety of model-related attacks against both image and text models in a data-free and black-box manner, achieving comparable performance to baseline methods that have access to the target models' training data and parameters in a white-box manner. This research serves as an important early warning to the community about the potential risks associated with generative AI-powered attacks on deep learning models.



## **13. Self-interpreting Adversarial Images**

cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2407.08970v3) [paper-pdf](http://arxiv.org/pdf/2407.08970v3)

**Authors**: Tingwei Zhang, Collin Zhang, John X. Morris, Eugene Bagdasarian, Vitaly Shmatikov

**Abstract**: We introduce a new type of indirect, cross-modal injection attacks against visual language models that enable creation of self-interpreting images. These images contain hidden "meta-instructions" that control how models answer users' questions about the image and steer their outputs to express an adversary-chosen style, sentiment, or point of view. Self-interpreting images act as soft prompts, conditioning the model to satisfy the adversary's (meta-)objective while still producing answers based on the image's visual content. Meta-instructions are thus a stronger form of prompt injection. Adversarial images look natural and the model's answers are coherent and plausible--yet they also follow the adversary-chosen interpretation, e.g., political spin, or even objectives that are not achievable with explicit text instructions. We evaluate the efficacy of self-interpreting images for a variety of models, interpretations, and user prompts. We describe how these attacks could cause harm by enabling creation of self-interpreting content that carries spam, misinformation, or spin. Finally, we discuss defenses.



## **14. Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs**

cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16534v1) [paper-pdf](http://arxiv.org/pdf/2501.16534v1)

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we present and evaluate a method to assess the robustness of LLM alignment. We observe that alignment embeds a safety classifier in the target model that is responsible for deciding between refusal and compliance. We seek to extract an approximation of this classifier, called a surrogate classifier, from the LLM. We develop an algorithm for identifying candidate classifiers from subsets of the LLM model. We evaluate the degree to which the candidate classifiers approximate the model's embedded classifier in benign (F1 score) and adversarial (using surrogates in a white-box attack) settings. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find attacks mounted on the surrogate models can be transferred with high accuracy. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70%, a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is a viable (and highly effective) means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks.



## **15. Smoothed Embeddings for Robust Language Models**

cs.LG

Presented in the Safe Generative AI Workshop at NeurIPS 2024

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16497v1) [paper-pdf](http://arxiv.org/pdf/2501.16497v1)

**Authors**: Ryo Hase, Md Rafi Ur Rashid, Ashley Lewis, Jing Liu, Toshiaki Koike-Akino, Kieran Parsons, Ye Wang

**Abstract**: Improving the safety and reliability of large language models (LLMs) is a crucial aspect of realizing trustworthy AI systems. Although alignment methods aim to suppress harmful content generation, LLMs are often still vulnerable to jailbreaking attacks that employ adversarial inputs that subvert alignment and induce harmful outputs. We propose the Randomized Embedding Smoothing and Token Aggregation (RESTA) defense, which adds random noise to the embedding vectors and performs aggregation during the generation of each output token, with the aim of better preserving semantic information. Our experiments demonstrate that our approach achieves superior robustness versus utility tradeoffs compared to the baseline defenses.



## **16. Towards Robust Stability Prediction in Smart Grids: GAN-based Approach under Data Constraints and Adversarial Challenges**

cs.CR

This work has been submitted to the IEEE Internet of Things Journal  for possible publication

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16490v1) [paper-pdf](http://arxiv.org/pdf/2501.16490v1)

**Authors**: Emad Efatinasab, Alessandro Brighente, Denis Donadel, Mauro Conti, Mirco Rampazzo

**Abstract**: Smart grids are critical for addressing the growing energy demand due to global population growth and urbanization. They enhance efficiency, reliability, and sustainability by integrating renewable energy. Ensuring their availability and safety requires advanced operational control and safety measures. Researchers employ AI and machine learning to assess grid stability, but challenges like the lack of datasets and cybersecurity threats, including adversarial attacks, persist. In particular, data scarcity is a key issue: obtaining grid instability instances is tough due to the need for significant expertise, resources, and time. However, they are essential to test novel research advancements and security mitigations. In this paper, we introduce a novel framework to detect instability in smart grids by employing only stable data. It relies on a Generative Adversarial Network (GAN) where the generator is trained to create instability data that are used along with stable data to train the discriminator. Moreover, we include a new adversarial training layer to improve robustness against adversarial attacks. Our solution, tested on a dataset composed of real-world stable and unstable samples, achieve accuracy up to 97.5\% in predicting grid stability and up to 98.9\% in detecting adversarial attacks. Moreover, we implemented our model in a single-board computer demonstrating efficient real-time decision-making with an average response time of less than 7ms. Our solution improves prediction accuracy and resilience while addressing data scarcity in smart grid management.



## **17. LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models**

cs.LG

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.15850v1) [paper-pdf](http://arxiv.org/pdf/2501.15850v1)

**Authors**: Yuewen Mei, Tong Nie, Jian Sun, Ye Tian

**Abstract**: Ensuring and improving the safety of autonomous driving systems (ADS) is crucial for the deployment of highly automated vehicles, especially in safety-critical events. To address the rarity issue, adversarial scenario generation methods are developed, in which behaviors of traffic participants are manipulated to induce safety-critical events. However, existing methods still face two limitations. First, identification of the adversarial participant directly impacts the effectiveness of the generation. However, the complexity of real-world scenarios, with numerous participants and diverse behaviors, makes identification challenging. Second, the potential of generated safety-critical scenarios to continuously improve ADS performance remains underexplored. To address these issues, we propose LLM-attacker: a closed-loop adversarial scenario generation framework leveraging large language models (LLMs). Specifically, multiple LLM agents are designed and coordinated to identify optimal attackers. Then, the trajectories of the attackers are optimized to generate adversarial scenarios. These scenarios are iteratively refined based on the performance of ADS, forming a feedback loop to improve ADS. Experimental results show that LLM-attacker can create more dangerous scenarios than other methods, and the ADS trained with it achieves a collision rate half that of training with normal scenarios. This indicates the ability of LLM-attacker to test and enhance the safety and robustness of ADS. Video demonstrations are provided at: https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.



## **18. Intelligent Code Embedding Framework for High-Precision Ransomware Detection via Multimodal Execution Path Analysis**

cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.15836v1) [paper-pdf](http://arxiv.org/pdf/2501.15836v1)

**Authors**: Levi Gareth, Maximilian Fairbrother, Peregrine Blackwood, Lucasta Underhill, Benedict Ruthermore

**Abstract**: Modern threat landscapes continue to evolve with increasing sophistication, challenging traditional detection methodologies and necessitating innovative solutions capable of addressing complex adversarial tactics. A novel framework was developed to identify ransomware activity through multimodal execution path analysis, integrating high-dimensional embeddings and dynamic heuristic derivation mechanisms to capture behavioral patterns across diverse attack variants. The approach demonstrated high adaptability, effectively mitigating obfuscation strategies and polymorphic characteristics often employed by ransomware families to evade detection. Comprehensive experimental evaluations revealed significant advancements in precision, recall, and accuracy metrics compared to baseline techniques, particularly under conditions of variable encryption speeds and obfuscated execution flows. The framework achieved scalable and computationally efficient performance, ensuring robust applicability across a range of system configurations, from resource-constrained environments to high-performance infrastructures. Notable findings included reduced false positive rates and enhanced detection latency, even for ransomware families employing sophisticated encryption mechanisms. The modular design allowed seamless integration of additional modalities, enabling extensibility and future-proofing against emerging threat vectors. Quantitative analyses further highlighted the system's energy efficiency, emphasizing its practicality for deployment in environments with stringent operational constraints. The results underline the importance of integrating advanced computational techniques and dynamic adaptability to safeguard digital ecosystems from increasingly complex threats.



## **19. A Privacy Model for Classical & Learned Bloom Filters**

cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.15751v1) [paper-pdf](http://arxiv.org/pdf/2501.15751v1)

**Authors**: Hayder Tirmazi

**Abstract**: The Classical Bloom Filter (CBF) is a class of Probabilistic Data Structures (PDS) for handling Approximate Query Membership (AMQ). The Learned Bloom Filter (LBF) is a recently proposed class of PDS that combines the Classical Bloom Filter with a Learning Model while preserving the Bloom Filter's one-sided error guarantees. Bloom Filters have been used in settings where inputs are sensitive and need to be private in the presence of an adversary with access to the Bloom Filter through an API or in the presence of an adversary who has access to the internal state of the Bloom Filter. Prior work has investigated the privacy of the Classical Bloom Filter providing attacks and defenses under various privacy definitions. In this work, we formulate a stronger differential privacy-based model for the Bloom Filter. We propose constructions of the Classical and Learned Bloom Filter that satisfy $(\epsilon, 0)$-differential privacy. This is also the first work that analyses and addresses the privacy of the Learned Bloom Filter under any rigorous model, which is an open problem.



## **20. Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings**

cs.LG

Accepted at the International Conference on Learning Representations  (ICLR) 2025. Code and pre-trained models are available at  https://github.com/AdaptiveMotorControlLab/AROS

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2410.10744v2) [paper-pdf](http://arxiv.org/pdf/2410.10744v2)

**Authors**: Hossein Mirzaei, Mackenzie W. Mathis

**Abstract**: Despite significant advancements in out-of-distribution (OOD) detection, existing methods still struggle to maintain robustness against adversarial attacks, compromising their reliability in critical real-world applications. Previous studies have attempted to address this challenge by exposing detectors to auxiliary OOD datasets alongside adversarial training. However, the increased data complexity inherent in adversarial training, and the myriad of ways that OOD samples can arise during testing, often prevent these approaches from establishing robust decision boundaries. To address these limitations, we propose AROS, a novel approach leveraging neural ordinary differential equations (NODEs) with Lyapunov stability theorem in order to obtain robust embeddings for OOD detection. By incorporating a tailored loss function, we apply Lyapunov stability theory to ensure that both in-distribution (ID) and OOD data converge to stable equilibrium points within the dynamical system. This approach encourages any perturbed input to return to its stable equilibrium, thereby enhancing the model's robustness against adversarial perturbations. To not use additional data, we generate fake OOD embeddings by sampling from low-likelihood regions of the ID data feature space, approximating the boundaries where OOD data are likely to reside. To then further enhance robustness, we propose the use of an orthogonal binary layer following the stable feature space, which maximizes the separation between the equilibrium points of ID and OOD samples. We validate our method through extensive experiments across several benchmarks, demonstrating superior performance, particularly under adversarial attacks. Notably, our approach improves robust detection performance from 37.8% to 80.1% on CIFAR-10 vs. CIFAR-100 and from 29.0% to 67.0% on CIFAR-100 vs. CIFAR-10.



## **21. FIT-Print: Towards False-claim-resistant Model Ownership Verification via Targeted Fingerprint**

cs.CR

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2501.15509v1) [paper-pdf](http://arxiv.org/pdf/2501.15509v1)

**Authors**: Shuo Shao, Haozhe Zhu, Hongwei Yao, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Model fingerprinting is a widely adopted approach to safeguard the intellectual property rights of open-source models by preventing their unauthorized reuse. It is promising and convenient since it does not necessitate modifying the protected model. In this paper, we revisit existing fingerprinting methods and reveal that they are vulnerable to false claim attacks where adversaries falsely assert ownership of any third-party model. We demonstrate that this vulnerability mostly stems from their untargeted nature, where they generally compare the outputs of given samples on different models instead of the similarities to specific references. Motivated by these findings, we propose a targeted fingerprinting paradigm (i.e., FIT-Print) to counteract false claim attacks. Specifically, FIT-Print transforms the fingerprint into a targeted signature via optimization. Building on the principles of FIT-Print, we develop bit-wise and list-wise black-box model fingerprinting methods, i.e., FIT-ModelDiff and FIT-LIME, which exploit the distance between model outputs and the feature attribution of specific samples as the fingerprint, respectively. Extensive experiments on benchmark models and datasets verify the effectiveness, conferrability, and resistance to false claim attacks of our FIT-Print.



## **22. Mitigating Spurious Negative Pairs for Robust Industrial Anomaly Detection**

cs.CV

Accepted at the 13th International Conference on Learning  Representations (ICLR) 2025

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2501.15434v1) [paper-pdf](http://arxiv.org/pdf/2501.15434v1)

**Authors**: Hossein Mirzaei, Mojtaba Nafez, Jafar Habibi, Mohammad Sabokrou, Mohammad Hossein Rohban

**Abstract**: Despite significant progress in Anomaly Detection (AD), the robustness of existing detection methods against adversarial attacks remains a challenge, compromising their reliability in critical real-world applications such as autonomous driving. This issue primarily arises from the AD setup, which assumes that training data is limited to a group of unlabeled normal samples, making the detectors vulnerable to adversarial anomaly samples during testing. Additionally, implementing adversarial training as a safeguard encounters difficulties, such as formulating an effective objective function without access to labels. An ideal objective function for adversarial training in AD should promote strong perturbations both within and between the normal and anomaly groups to maximize margin between normal and anomaly distribution. To address these issues, we first propose crafting a pseudo-anomaly group derived from normal group samples. Then, we demonstrate that adversarial training with contrastive loss could serve as an ideal objective function, as it creates both inter- and intra-group perturbations. However, we notice that spurious negative pairs compromise the conventional contrastive loss to achieve robust AD. Spurious negative pairs are those that should be closely mapped but are erroneously separated. These pairs introduce noise and misguide the direction of inter-group adversarial perturbations. To overcome the effect of spurious negative pairs, we define opposite pairs and adversarially pull them apart to strengthen inter-group perturbations. Experimental results demonstrate our superior performance in both clean and adversarial scenarios, with a 26.1% improvement in robust detection across various challenging benchmark datasets. The implementation of our work is available at: https://github.com/rohban-lab/COBRA.



## **23. MAPPING: Debiasing Graph Neural Networks for Fair Node Classification with Limited Sensitive Information Leakage**

cs.LG

Accepted by WWW Journal. Code is available at  https://github.com/yings0930/MAPPING

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2401.12824v2) [paper-pdf](http://arxiv.org/pdf/2401.12824v2)

**Authors**: Ying Song, Balaji Palanisamy

**Abstract**: Despite remarkable success in diverse web-based applications, Graph Neural Networks(GNNs) inherit and further exacerbate historical discrimination and social stereotypes, which critically hinder their deployments in high-stake domains such as online clinical diagnosis, financial crediting, etc. However, current fairness research that primarily craft on i.i.d data, cannot be trivially replicated to non-i.i.d. graph structures with topological dependence among samples. Existing fair graph learning typically favors pairwise constraints to achieve fairness but fails to cast off dimensional limitations and generalize them into multiple sensitive attributes; besides, most studies focus on in-processing techniques to enforce and calibrate fairness, constructing a model-agnostic debiasing GNN framework at the pre-processing stage to prevent downstream misuses and improve training reliability is still largely under-explored. Furthermore, previous work on GNNs tend to enhance either fairness or privacy individually but few probe into their interplays. In this paper, we propose a novel model-agnostic debiasing framework named MAPPING (\underline{M}asking \underline{A}nd \underline{P}runing and Message-\underline{P}assing train\underline{ING}) for fair node classification, in which we adopt the distance covariance($dCov$)-based fairness constraints to simultaneously reduce feature and topology biases in arbitrary dimensions, and combine them with adversarial debiasing to confine the risks of attribute inference attacks. Experiments on real-world datasets with different GNN variants demonstrate the effectiveness and flexibility of MAPPING. Our results show that MAPPING can achieve better trade-offs between utility and fairness, and mitigate privacy risks of sensitive information leakage.



## **24. Hiding in Plain Sight: An IoT Traffic Camouflage Framework for Enhanced Privacy**

cs.CR

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2501.15395v1) [paper-pdf](http://arxiv.org/pdf/2501.15395v1)

**Authors**: Daniel Adu Worae, Spyridon Mastorakis

**Abstract**: The rapid growth of Internet of Things (IoT) devices has introduced significant challenges to privacy, particularly as network traffic analysis techniques evolve. While encryption protects data content, traffic attributes such as packet size and timing can reveal sensitive information about users and devices. Existing single-technique obfuscation methods, such as packet padding, often fall short in dynamic environments like smart homes due to their predictability, making them vulnerable to machine learning-based attacks. This paper introduces a multi-technique obfuscation framework designed to enhance privacy by disrupting traffic analysis. The framework leverages six techniques-Padding, Padding with XORing, Padding with Shifting, Constant Size Padding, Fragmentation, and Delay Randomization-to obscure traffic patterns effectively. Evaluations on three public datasets demonstrate significant reductions in classifier performance metrics, including accuracy, precision, recall, and F1 score. We assess the framework's robustness against adversarial tactics by retraining and fine-tuning neural network classifiers on obfuscated traffic. The results reveal a notable degradation in classifier performance, underscoring the framework's resilience against adaptive attacks. Furthermore, we evaluate communication and system performance, showing that higher obfuscation levels enhance privacy but may increase latency and communication overhead.



## **25. AI-Driven Secure Data Sharing: A Trustworthy and Privacy-Preserving Approach**

cs.CR

6 pages, 4 figures

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2501.15363v1) [paper-pdf](http://arxiv.org/pdf/2501.15363v1)

**Authors**: Al Amin, Kamrul Hasan, Sharif Ullah, Liang Hong

**Abstract**: In the era of data-driven decision-making, ensuring the privacy and security of shared data is paramount across various domains. Applying existing deep neural networks (DNNs) to encrypted data is critical and often compromises performance, security, and computational overhead. To address these limitations, this research introduces a secure framework consisting of a learnable encryption method based on the block-pixel operation to encrypt the data and subsequently integrate it with the Vision Transformer (ViT). The proposed framework ensures data privacy and security by creating unique scrambling patterns per key, providing robust performance against adversarial attacks without compromising computational efficiency and data integrity. The framework was tested on sensitive medical datasets to validate its efficacy, proving its ability to handle highly confidential information securely. The suggested framework was validated with a 94\% success rate after extensive testing on real-world datasets, such as MRI brain tumors and histological scans of lung and colon cancers. Additionally, the framework was tested under diverse adversarial attempts against secure data sharing with optimum performance and demonstrated its effectiveness in various threat scenarios. These comprehensive analyses underscore its robustness, making it a trustworthy solution for secure data sharing in critical applications.



## **26. A theoretical basis for MEV**

cs.CR

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2302.02154v4) [paper-pdf](http://arxiv.org/pdf/2302.02154v4)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Maximal Extractable Value (MEV) refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream DeFi protocols are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the increasing real-world impact of these attacks, their theoretical foundations remain insufficiently established. We propose a formal theory of MEV, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against MEV attacks.



## **27. Killing it with Zero-Shot: Adversarially Robust Novelty Detection**

cs.LG

Accepted to the Proceedings of the IEEE International Conference on  Acoustics, Speech, and Signal Processing (ICASSP) 2024

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15271v1) [paper-pdf](http://arxiv.org/pdf/2501.15271v1)

**Authors**: Hossein Mirzaei, Mohammad Jafari, Hamid Reza Dehbashi, Zeinab Sadat Taghavi, Mohammad Sabokrou, Mohammad Hossein Rohban

**Abstract**: Novelty Detection (ND) plays a crucial role in machine learning by identifying new or unseen data during model inference. This capability is especially important for the safe and reliable operation of automated systems. Despite advances in this field, existing techniques often fail to maintain their performance when subject to adversarial attacks. Our research addresses this gap by marrying the merits of nearest-neighbor algorithms with robust features obtained from models pretrained on ImageNet. We focus on enhancing the robustness and performance of ND algorithms. Experimental results demonstrate that our approach significantly outperforms current state-of-the-art methods across various benchmarks, particularly under adversarial conditions. By incorporating robust pretrained features into the k-NN algorithm, we establish a new standard for performance and robustness in the field of robust ND. This work opens up new avenues for research aimed at fortifying machine learning systems against adversarial vulnerabilities. Our implementation is publicly available at https://github.com/rohban-lab/ZARND.



## **28. Mirage in the Eyes: Hallucination Attack on Multi-modal Large Language Models with Only Attention Sink**

cs.LG

USENIX Security 2025

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15269v1) [paper-pdf](http://arxiv.org/pdf/2501.15269v1)

**Authors**: Yining Wang, Mi Zhang, Junjie Sun, Chenyue Wang, Min Yang, Hui Xue, Jialing Tao, Ranjie Duan, Jiexi Liu

**Abstract**: Fusing visual understanding into language generation, Multi-modal Large Language Models (MLLMs) are revolutionizing visual-language applications. Yet, these models are often plagued by the hallucination problem, which involves generating inaccurate objects, attributes, and relationships that do not match the visual content. In this work, we delve into the internal attention mechanisms of MLLMs to reveal the underlying causes of hallucination, exposing the inherent vulnerabilities in the instruction-tuning process.   We propose a novel hallucination attack against MLLMs that exploits attention sink behaviors to trigger hallucinated content with minimal image-text relevance, posing a significant threat to critical downstream applications. Distinguished from previous adversarial methods that rely on fixed patterns, our approach generates dynamic, effective, and highly transferable visual adversarial inputs, without sacrificing the quality of model responses. Comprehensive experiments on 6 prominent MLLMs demonstrate the efficacy of our attack in compromising black-box MLLMs even with extensive mitigating mechanisms, as well as the promising results against cutting-edge commercial APIs, such as GPT-4o and Gemini 1.5. Our code is available at https://huggingface.co/RachelHGF/Mirage-in-the-Eyes.



## **29. Pre-trained Model Guided Mixture Knowledge Distillation for Adversarial Federated Learning**

cs.CV

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15257v1) [paper-pdf](http://arxiv.org/pdf/2501.15257v1)

**Authors**: Yu Qiao, Huy Q. Le, Apurba Adhikary, Choong Seon Hong

**Abstract**: This paper aims to improve the robustness of a small global model while maintaining clean accuracy under adversarial attacks and non-IID challenges in federated learning. By leveraging the concise knowledge embedded in the class probabilities from a pre-trained model for both clean and adversarial image classification, we propose a Pre-trained Model-guided Adversarial Federated Learning (PM-AFL) training paradigm. This paradigm integrates vanilla mixture and adversarial mixture knowledge distillation to effectively balance accuracy and robustness while promoting local models to learn from diverse data. Specifically, for clean accuracy, we adopt a dual distillation strategy where the class probabilities of randomly paired images and their blended versions are aligned between the teacher model and the local models. For adversarial robustness, we use a similar distillation approach but replace clean samples on the local side with adversarial examples. Moreover, considering the bias between local and global models, we also incorporate a consistency regularization term to ensure that local adversarial predictions stay aligned with their corresponding global clean ones. These strategies collectively enable local models to absorb diverse knowledge from the teacher model while maintaining close alignment with the global model, thereby mitigating overfitting to local optima and enhancing the generalization of the global model. Experiments demonstrate that the PM-AFL-based paradigm outperforms other methods that integrate defense strategies by a notable margin.



## **30. Comprehensive Evaluation of Cloaking Backdoor Attacks on Object Detector in Real-World**

cs.CR

arXiv admin note: text overlap with arXiv:2201.08619

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15101v1) [paper-pdf](http://arxiv.org/pdf/2501.15101v1)

**Authors**: Hua Ma, Alsharif Abuadbba, Yansong Gao, Hyoungshick Kim, Surya Nepal

**Abstract**: The exploration of backdoor vulnerabilities in object detectors, particularly in real-world scenarios, remains limited. A significant challenge lies in the absence of a natural physical backdoor dataset, and constructing such a dataset is both time- and labor-intensive. In this work, we address this gap by creating a large-scale dataset comprising approximately 11,800 images/frames with annotations featuring natural objects (e.g., T-shirts and hats) as triggers to incur cloaking adversarial effects in diverse real-world scenarios. This dataset is tailored for the study of physical backdoors in object detectors. Leveraging this dataset, we conduct a comprehensive evaluation of an insidious cloaking backdoor effect against object detectors, wherein the bounding box around a person vanishes when the individual is near a natural object (e.g., a commonly available T-shirt) in front of the detector. Our evaluations encompass three prevalent attack surfaces: data outsourcing, model outsourcing, and the use of pretrained models. The cloaking effect is successfully implanted in object detectors across all three attack surfaces. We extensively evaluate four popular object detection algorithms (anchor-based Yolo-V3, Yolo-V4, Faster R-CNN, and anchor-free CenterNet) using 19 videos (totaling approximately 11,800 frames) in real-world scenarios. Our results demonstrate that the backdoor attack exhibits remarkable robustness against various factors, including movement, distance, angle, non-rigid deformation, and lighting. In data and model outsourcing scenarios, the attack success rate (ASR) in most videos reaches 100% or near it, while the clean data accuracy of the backdoored model remains indistinguishable from that of the clean model, making it impossible to detect backdoor behavior through a validation set.



## **31. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Proactive Adversarial Defense**

cs.CR

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2407.15524v5) [paper-pdf](http://arxiv.org/pdf/2407.15524v5)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy due to its sensitivity to adversarial attacks. Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, existing anti-adversarial methods typically counteract adversarial perturbations post-attack, while we have devised a proactive strategy that preempts by safeguarding media upfront, effectively neutralizing potential adversarial effects before the third-party attacks occur. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first, to our knowledge, effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks.



## **32. Poisoning Prevention in Federated Learning and Differential Privacy via Stateful Proofs of Execution**

cs.CR

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2404.06721v4) [paper-pdf](http://arxiv.org/pdf/2404.06721v4)

**Authors**: Norrathep Rattanavipanon, Ivan De Oliveira Nunes

**Abstract**: The rise in IoT-driven distributed data analytics, coupled with increasing privacy concerns, has led to a demand for effective privacy-preserving and federated data collection/model training mechanisms. In response, approaches such as Federated Learning (FL) and Local Differential Privacy (LDP) have been proposed and attracted much attention over the past few years. However, they still share the common limitation of being vulnerable to poisoning attacks wherein adversaries compromising edge devices feed forged (a.k.a. poisoned) data to aggregation back-ends, undermining the integrity of FL/LDP results.   In this work, we propose a system-level approach to remedy this issue based on a novel security notion of Proofs of Stateful Execution (PoSX) for IoT/embedded devices' software. To realize the PoSX concept, we design SLAPP: a System-Level Approach for Poisoning Prevention. SLAPP leverages commodity security features of embedded devices - in particular ARM TrustZoneM security extensions - to verifiably bind raw sensed data to their correct usage as part of FL/LDP edge device routines. As a consequence, it offers robust security guarantees against poisoning. Our evaluation, based on real-world prototypes featuring multiple cryptographic primitives and data collection schemes, showcases SLAPP's security and low overhead.



## **33. VideoPure: Diffusion-based Adversarial Purification for Video Recognition**

cs.CV

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.14999v1) [paper-pdf](http://arxiv.org/pdf/2501.14999v1)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Jiyuan Fu, Lingyi Hong, Jinglun Li, Wenqiang Zhang

**Abstract**: Recent work indicates that video recognition models are vulnerable to adversarial examples, posing a serious security risk to downstream applications. However, current research has primarily focused on adversarial attacks, with limited work exploring defense mechanisms. Furthermore, due to the spatial-temporal complexity of videos, existing video defense methods face issues of high cost, overfitting, and limited defense performance. Recently, diffusion-based adversarial purification methods have achieved robust defense performance in the image domain. However, due to the additional temporal dimension in videos, directly applying these diffusion-based adversarial purification methods to the video domain suffers performance and efficiency degradation. To achieve an efficient and effective video adversarial defense method, we propose the first diffusion-based video purification framework to improve video recognition models' adversarial robustness: VideoPure. Given an adversarial example, we first employ temporal DDIM inversion to transform the input distribution into a temporally consistent and trajectory-defined distribution, covering adversarial noise while preserving more video structure. Then, during DDIM denoising, we leverage intermediate results at each denoising step and conduct guided spatial-temporal optimization, removing adversarial noise while maintaining temporal consistency. Finally, we input the list of optimized intermediate results into the video recognition model for multi-step voting to obtain the predicted class. We investigate the defense performance of our method against black-box, gray-box, and adaptive attacks on benchmark datasets and models. Compared with other adversarial purification methods, our method overall demonstrates better defense performance against different attacks. Our code is available at https://github.com/deep-kaixun/VideoPure.



## **34. Untelegraphable Encryption and its Applications**

quant-ph

56 pages

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2410.24189v2) [paper-pdf](http://arxiv.org/pdf/2410.24189v2)

**Authors**: Jeffrey Champion, Fuyuki Kitagawa, Ryo Nishimaki, Takashi Yamakawa

**Abstract**: We initiate the study of untelegraphable encryption (UTE), founded on the no-telegraphing principle, which allows an encryptor to encrypt a message such that a binary string representation of the ciphertext cannot be decrypted by a user with the secret key, a task that is classically impossible. This is a natural relaxation of unclonable encryption (UE), inspired by the recent work of Nehoran and Zhandry (ITCS 2024), who showed a computational separation between the no-cloning and no-telegraphing principles. In this work, we define and construct UTE information-theoretically in the plain model. Building off this, we give several applications of UTE and study the interplay of UTE with UE and well-studied tasks in quantum state learning, yielding the following contributions:   - A construction of collusion-resistant UTE from plain secret-key encryption, which we then show denies the existence of hyper-efficient shadow tomography (HEST). By building a relaxation of collusion-resistant UTE, we show the impossibility of HEST assuming only pseudorandom state generators (which may not imply one-way functions). This almost unconditionally answers an open inquiry of Aaronson (STOC 2018).   - A construction of UTE from a one-shot message authentication code in the classical oracle model, such that there is an explicit attack that breaks UE security for an unbounded polynomial number of decryptors.   - A construction of everlasting secure collusion-resistant UTE, where the decryptor adversary can run in unbounded time, in the quantum random oracle model (QROM), and formal evidence that a construction in the plain model is a challenging task. We leverage this construction to show that HEST with unbounded post-processing time is impossible in the QROM.   - Constructions of secret sharing resilient to joint and unbounded classical leakage and untelegraphable functional encryption.



## **35. PANTS: Practical Adversarial Network Traffic Samples against ML-powered Networking Classifiers**

cs.CR

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2409.04691v2) [paper-pdf](http://arxiv.org/pdf/2409.04691v2)

**Authors**: Minhao Jin, Maria Apostolaki

**Abstract**: Multiple network management tasks, from resource allocation to intrusion detection, rely on some form of ML-based network traffic classification (MNC). Despite their potential, MNCs are vulnerable to adversarial inputs, which can lead to outages, poor decision-making, and security violations, among other issues. The goal of this paper is to help network operators assess and enhance the robustness of their MNC against adversarial inputs. The most critical step for this is generating inputs that can fool the MNC while being realizable under various threat models. Compared to other ML models, finding adversarial inputs against MNCs is more challenging due to the existence of non-differentiable components e.g., traffic engineering and the need to constrain inputs to preserve semantics and ensure reliability. These factors prevent the direct use of well-established gradient-based methods developed in adversarial ML (AML). To address these challenges, we introduce PANTS, a practical white-box framework that uniquely integrates AML techniques with Satisfiability Modulo Theories (SMT) solvers to generate adversarial inputs for MNCs. We also embed PANTS into an iterative adversarial training process that enhances the robustness of MNCs against adversarial inputs. PANTS is 70% and 2x more likely in median to find adversarial inputs against target MNCs compared to state-of-the-art baselines, namely Amoeba and BAP. PANTS improves the robustness of the target MNCs by 52.7% (even against attackers outside of what is considered during robustification) without sacrificing their accuracy.



## **36. Towards Scalable Topological Regularizers**

cs.LG

31 pages, accepted to ICLR 2025

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14641v1) [paper-pdf](http://arxiv.org/pdf/2501.14641v1)

**Authors**: Hiu-Tung Wong, Darrick Lee, Hong Yan

**Abstract**: Latent space matching, which consists of matching distributions of features in latent space, is a crucial component for tasks such as adversarial attacks and defenses, domain adaptation, and generative modelling. Metrics for probability measures, such as Wasserstein and maximum mean discrepancy, are commonly used to quantify the differences between such distributions. However, these are often costly to compute, or do not appropriately take the geometric and topological features of the distributions into consideration. Persistent homology is a tool from topological data analysis which quantifies the multi-scale topological structure of point clouds, and has recently been used as a topological regularizer in learning tasks. However, computation costs preclude larger scale computations, and discontinuities in the gradient lead to unstable training behavior such as in adversarial tasks. We propose the use of principal persistence measures, based on computing the persistent homology of a large number of small subsamples, as a topological regularizer. We provide a parallelized GPU implementation of this regularizer, and prove that gradients are continuous for smooth densities. Furthermore, we demonstrate the efficacy of this regularizer on shape matching, image generation, and semi-supervised learning tasks, opening the door towards a scalable regularizer for topological features.



## **37. Self-playing Adversarial Language Game Enhances LLM Reasoning**

cs.CL

Accepted by NeurIPS 2024

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2404.10642v3) [paper-pdf](http://arxiv.org/pdf/2404.10642v3)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Zheng Yuan, Yong Dai, Lei Han, Nan Du, Xiaolong Li

**Abstract**: We explore the potential of self-play training for large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players must have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by Self-Playing this Adversarial language Game (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs' reasoning abilities. The code is available at https://github.com/Linear95/SPAG.



## **38. Real-world Edge Neural Network Implementations Leak Private Interactions Through Physical Side Channel**

cs.CR

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14512v1) [paper-pdf](http://arxiv.org/pdf/2501.14512v1)

**Authors**: Zhuoran Liu, Senna van Hoek, Péter Horváth, Dirk Lauret, Xiaoyun Xu, Lejla Batina

**Abstract**: Neural networks have become a fundamental component of numerous practical applications, and their implementations, which are often accelerated by hardware, are integrated into all types of real-world physical devices. User interactions with neural networks on hardware accelerators are commonly considered privacy-sensitive. Substantial efforts have been made to uncover vulnerabilities and enhance privacy protection at the level of machine learning algorithms, including membership inference attacks, differential privacy, and federated learning. However, neural networks are ultimately implemented and deployed on physical devices, and current research pays comparatively less attention to privacy protection at the implementation level. In this paper, we introduce a generic physical side-channel attack, ScaAR, that extracts user interactions with neural networks by leveraging electromagnetic (EM) emissions of physical devices. Our proposed attack is implementation-agnostic, meaning it does not require the adversary to possess detailed knowledge of the hardware or software implementations, thanks to the capabilities of deep learning-based side-channel analysis (DLSCA). Experimental results demonstrate that, through the EM side channel, ScaAR can effectively extract the class label of user interactions with neural classifiers, including inputs and outputs, on the AMD-Xilinx MPSoC ZCU104 FPGA and Raspberry Pi 3 B. In addition, for the first time, we provide side-channel analysis on edge Large Language Model (LLM) implementations on the Raspberry Pi 5, showing that EM side channel leaks interaction data, and different LLM tokens can be distinguishable from the EM traces.



## **39. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

Accepted by ICLR2025

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2406.18849v3) [paper-pdf](http://arxiv.org/pdf/2406.18849v3)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released at \url{https://github.com/Robin-WZQ/Dysca}.



## **40. A Note on Implementation Errors in Recent Adaptive Attacks Against Multi-Resolution Self-Ensembles**

cs.CR

4 pages, 2 figures, technical note addressing an issue in  arXiv:2411.14834v1

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14496v1) [paper-pdf](http://arxiv.org/pdf/2501.14496v1)

**Authors**: Stanislav Fort

**Abstract**: This note documents an implementation issue in recent adaptive attacks (Zhang et al. [2024]) against the multi-resolution self-ensemble defense (Fort and Lakshminarayanan [2024]). The implementation allowed adversarial perturbations to exceed the standard $L_\infty = 8/255$ bound by up to a factor of 20$\times$, reaching magnitudes of up to $L_\infty = 160/255$. When attacks are properly constrained within the intended bounds, the defense maintains non-trivial robustness. Beyond highlighting the importance of careful validation in adversarial machine learning research, our analysis reveals an intriguing finding: properly bounded adaptive attacks against strong multi-resolution self-ensembles often align with human perception, suggesting the need to reconsider how we measure adversarial robustness.



## **41. Optimal Strategies for Federated Learning Maintaining Client Privacy**

cs.LG

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14453v1) [paper-pdf](http://arxiv.org/pdf/2501.14453v1)

**Authors**: Uday Bhaskar, Varul Srivastava, Avyukta Manjunatha Vummintala, Naresh Manwani, Sujit Gujar

**Abstract**: Federated Learning (FL) emerged as a learning method to enable the server to train models over data distributed among various clients. These clients are protective about their data being leaked to the server, any other client, or an external adversary, and hence, locally train the model and share it with the server rather than sharing the data. The introduction of sophisticated inferencing attacks enabled the leakage of information about data through access to model parameters. To tackle this challenge, privacy-preserving federated learning aims to achieve differential privacy through learning algorithms like DP-SGD. However, such methods involve adding noise to the model, data, or gradients, reducing the model's performance.   This work provides a theoretical analysis of the tradeoff between model performance and communication complexity of the FL system. We formally prove that training for one local epoch per global round of training gives optimal performance while preserving the same privacy budget. We also investigate the change of utility (tied to privacy) of FL models with a change in the number of clients and observe that when clients are training using DP-SGD and argue that for the same privacy budget, the utility improved with increased clients. We validate our findings through experiments on real-world datasets. The results from this paper aim to improve the performance of privacy-preserving federated learning systems.



## **42. Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors**

cs.CL

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14250v1) [paper-pdf](http://arxiv.org/pdf/2501.14250v1)

**Authors**: Yi Zhao, Youzhi Zhang

**Abstract**: Large language models (LLMs) are widely used in real-world applications, raising concerns about their safety and trustworthiness. While red-teaming with jailbreak prompts exposes the vulnerabilities of LLMs, current efforts focus primarily on single-turn attacks, overlooking the multi-turn strategies used by real-world adversaries. Existing multi-turn methods rely on static patterns or predefined logical chains, failing to account for the dynamic strategies during attacks. We propose Siren, a learning-based multi-turn attack framework designed to simulate real-world human jailbreak behaviors. Siren consists of three stages: (1) training set construction utilizing Turn-Level LLM feedback (Turn-MF), (2) post-training attackers with supervised fine-tuning (SFT) and direct preference optimization (DPO), and (3) interactions between the attacking and target LLMs. Experiments demonstrate that Siren achieves an attack success rate (ASR) of 90% with LLaMA-3-8B as the attacker against Gemini-1.5-Pro as the target model, and 70% with Mistral-7B against GPT-4o, significantly outperforming single-turn baselines. Moreover, Siren with a 7B-scale model achieves performance comparable to a multi-turn baseline that leverages GPT-4o as the attacker, while requiring fewer turns and employing decomposition strategies that are better semantically aligned with attack goals. We hope Siren inspires the development of stronger defenses against advanced multi-turn jailbreak attacks under realistic scenarios. Code is available at https://github.com/YiyiyiZhao/siren. Warning: This paper contains potentially harmful text.



## **43. GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm**

cs.CV

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14230v1) [paper-pdf](http://arxiv.org/pdf/2501.14230v1)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Christopher Leckie, Isao Echizen

**Abstract**: A critical requirement for deep learning models is ensuring their robustness against adversarial attacks. These attacks commonly introduce noticeable perturbations, compromising the visual fidelity of adversarial examples. Another key challenge is that while white-box algorithms can generate effective adversarial perturbations, they require access to the model gradients, limiting their practicality in many real-world scenarios. Existing attack mechanisms struggle to achieve similar efficacy without access to these gradients. In this paper, we introduce GreedyPixel, a novel pixel-wise greedy algorithm designed to generate high-quality adversarial examples using only query-based feedback from the target model. GreedyPixel improves computational efficiency in what is typically a brute-force process by perturbing individual pixels in sequence, guided by a pixel-wise priority map. This priority map is constructed by ranking gradients obtained from a surrogate model, providing a structured path for perturbation. Our results demonstrate that GreedyPixel achieves attack success rates comparable to white-box methods without the need for gradient information, and surpasses existing algorithms in black-box settings, offering higher success rates, reduced computational time, and imperceptible perturbations. These findings underscore the advantages of GreedyPixel in terms of attack efficacy, time efficiency, and visual quality.



## **44. Reinforcement Learning Platform for Adversarial Black-box Attacks with Custom Distortion Filters**

cs.LG

Under Review for 2025 AAAI Conference on Artificial Intelligence  Proceedings

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.14122v1) [paper-pdf](http://arxiv.org/pdf/2501.14122v1)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Ricardo Luna Gutierrez, Antonio Guillen

**Abstract**: We present a Reinforcement Learning Platform for Adversarial Black-box untargeted and targeted attacks, RLAB, that allows users to select from various distortion filters to create adversarial examples. The platform uses a Reinforcement Learning agent to add minimum distortion to input images while still causing misclassification by the target model. The agent uses a novel dual-action method to explore the input image at each step to identify sensitive regions for adding distortions while removing noises that have less impact on the target model. This dual action leads to faster and more efficient convergence of the attack. The platform can also be used to measure the robustness of image classification models against specific distortion types. Also, retraining the model with adversarial samples significantly improved robustness when evaluated on benchmark datasets. The proposed platform outperforms state-of-the-art methods in terms of the average number of queries required to cause misclassification. This advances trustworthiness with a positive social impact.



## **45. Noisy Data Meets Privacy: Training Local Models with Post-Processed Remote Queries**

cs.LG

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2405.16361v2) [paper-pdf](http://arxiv.org/pdf/2405.16361v2)

**Authors**: Kexin Li, Aastha Mehta, David Lie

**Abstract**: The adoption of large cloud-based models for inference in privacy-sensitive domains, such as homeless care systems and medical imaging, raises concerns about end-user data privacy. A common solution is adding locally differentially private (LDP) noise to queries before transmission, but this often reduces utility. LDPKiT, which stands for Local Differentially-Private and Utility-Preserving Inference via Knowledge Transfer, addresses the concern by generating a privacy-preserving inference dataset aligned with the private data distribution. This dataset is used to train a reliable local model for inference on sensitive inputs. LDPKiT employs a two-layer noise injection framework that leverages LDP and its post-processing property to create a privacy-protected inference dataset. The first layer ensures privacy, while the second layer helps to recover utility by creating a sufficiently large dataset for subsequent local model extraction using noisy labels returned from a cloud model on privacy-protected noisy inputs. Our experiments on Fashion-MNIST, SVHN and PathMNIST medical datasets demonstrate that LDPKiT effectively improves utility while preserving privacy. Moreover, the benefits of using LDPKiT increase at higher, more privacy-protective noise levels. For instance, on SVHN, LDPKiT achieves similar inference accuracy with $\epsilon=1.25$ as it does with $\epsilon=2.0$, providing stronger privacy guarantees with less than a 2% drop in accuracy. Furthermore, we perform extensive sensitivity analyses to evaluate the impact of dataset sizes on LDPKiT's effectiveness and systematically analyze the latent space representations to offer a theoretical explanation for its accuracy improvements. Lastly, we qualitatively and quantitatively demonstrate that the type of knowledge distillation performed by LDPKiT is ethical and fundamentally distinct from adversarial model extraction attacks.



## **46. Defending against Adversarial Malware Attacks on ML-based Android Malware Detection Systems**

cs.CR

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.13782v1) [paper-pdf](http://arxiv.org/pdf/2501.13782v1)

**Authors**: Ping He, Lorenzo Cavallaro, Shouling Ji

**Abstract**: Android malware presents a persistent threat to users' privacy and data integrity. To combat this, researchers have proposed machine learning-based (ML-based) Android malware detection (AMD) systems. However, adversarial Android malware attacks compromise the detection integrity of the ML-based AMD systems, raising significant concerns. Existing defenses against adversarial Android malware provide protections against feature space attacks which generate adversarial feature vectors only, leaving protection against realistic threats from problem space attacks which generate real adversarial malware an open problem. In this paper, we address this gap by proposing ADD, a practical adversarial Android malware defense framework designed as a plug-in to enhance the adversarial robustness of the ML-based AMD systems against problem space attacks. Our extensive evaluation across various ML-based AMD systems demonstrates that ADD is effective against state-of-the-art problem space adversarial Android malware attacks. Additionally, ADD shows the defense effectiveness in enhancing the adversarial robustness of real-world antivirus solutions.



## **47. Crossfire: An Elastic Defense Framework for Graph Neural Networks Under Bit Flip Attacks**

cs.LG

Accepted at AAAI 2025, DOI will be included after publication

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.13776v1) [paper-pdf](http://arxiv.org/pdf/2501.13776v1)

**Authors**: Lorenz Kummer, Samir Moustafa, Wilfried Gansterer, Nils Kriege

**Abstract**: Bit Flip Attacks (BFAs) are a well-established class of adversarial attacks, originally developed for Convolutional Neural Networks within the computer vision domain. Most recently, these attacks have been extended to target Graph Neural Networks (GNNs), revealing significant vulnerabilities. This new development naturally raises questions about the best strategies to defend GNNs against BFAs, a challenge for which no solutions currently exist. Given the applications of GNNs in critical fields, any defense mechanism must not only maintain network performance, but also verifiably restore the network to its pre-attack state. Verifiably restoring the network to its pre-attack state also eliminates the need for costly evaluations on test data to ensure network quality. We offer first insights into the effectiveness of existing honeypot- and hashing-based defenses against BFAs adapted from the computer vision domain to GNNs, and characterize the shortcomings of these approaches. To overcome their limitations, we propose Crossfire, a hybrid approach that exploits weight sparsity and combines hashing and honeypots with bit-level correction of out-of-distribution weight elements to restore network integrity. Crossfire is retraining-free and does not require labeled data. Averaged over 2,160 experiments on six benchmark datasets, Crossfire offers a 21.8% higher probability than its competitors of reconstructing a GNN attacked by a BFA to its pre-attack state. These experiments cover up to 55 bit flips from various attacks. Moreover, it improves post-repair prediction quality by 10.85%. Computational and storage overheads are negligible compared to the inherent complexity of even the simplest GNNs.



## **48. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

cs.CR

Accepted at AAAI 2025 (Oral)

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2408.10738v3) [paper-pdf](http://arxiv.org/pdf/2408.10738v3)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.



## **49. Device-aware Optical Adversarial Attack for a Portable Projector-camera System**

cs.CV

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.14005v1) [paper-pdf](http://arxiv.org/pdf/2501.14005v1)

**Authors**: Ning Jiang, Yanhong Liu, Dingheng Zeng, Yue Feng, Weihong Deng, Ying Li

**Abstract**: Deep-learning-based face recognition (FR) systems are susceptible to adversarial examples in both digital and physical domains. Physical attacks present a greater threat to deployed systems as adversaries can easily access the input channel, allowing them to provide malicious inputs to impersonate a victim. This paper addresses the limitations of existing projector-camera-based adversarial light attacks in practical FR setups. By incorporating device-aware adaptations into the digital attack algorithm, such as resolution-aware and color-aware adjustments, we mitigate the degradation from digital to physical domains. Experimental validation showcases the efficacy of our proposed algorithm against real and spoof adversaries, achieving high physical similarity scores in FR models and state-of-the-art commercial systems. On average, there is only a 14% reduction in scores from digital to physical attacks, with high attack success rate in both white- and black-box scenarios.



## **50. Black-Box Adversarial Attack on Vision Language Models for Autonomous Driving**

cs.CV

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.13563v1) [paper-pdf](http://arxiv.org/pdf/2501.13563v1)

**Authors**: Lu Wang, Tianyuan Zhang, Yang Qu, Siyuan Liang, Yuwei Chen, Aishan Liu, Xianglong Liu, Dacheng Tao

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities; however, these models remain highly susceptible to adversarial attacks. While existing research has explored white-box attacks to some extent, the more practical and challenging black-box scenarios remain largely underexplored due to their inherent difficulty. In this paper, we take the first step toward designing black-box adversarial attacks specifically targeting VLMs in AD. We identify two key challenges for achieving effective black-box attacks in this context: the effectiveness across driving reasoning chains in AD systems and the dynamic nature of driving scenarios. To address this, we propose Cascading Adversarial Disruption (CAD). It first introduces Decision Chain Disruption, which targets low-level reasoning breakdown by generating and injecting deceptive semantics, ensuring the perturbations remain effective across the entire decision-making chain. Building on this, we present Risky Scene Induction, which addresses dynamic adaptation by leveraging a surrogate VLM to understand and construct high-level risky scenarios that are likely to result in critical errors in the current driving contexts. Extensive experiments conducted on multiple AD VLMs and benchmarks demonstrate that CAD achieves state-of-the-art attack effectiveness, significantly outperforming existing methods (+13.43% on average). Moreover, we validate its practical applicability through real-world attacks on AD vehicles powered by VLMs, where the route completion rate drops by 61.11% and the vehicle crashes directly into the obstacle vehicle with adversarial patches. Finally, we release CADA dataset, comprising 18,808 adversarial visual-question-answer pairs, to facilitate further evaluation and research in this critical domain. Our codes and dataset will be available after paper's acceptance.



