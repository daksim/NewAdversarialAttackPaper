# Latest Adversarial Attack Papers
**update at 2023-09-05 10:56:40**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Baseline Defenses for Adversarial Attacks Against Aligned Language Models**

cs.LG

12 pages

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2309.00614v1) [paper-pdf](http://arxiv.org/pdf/2309.00614v1)

**Authors**: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein

**Abstract**: As Large Language Models quickly become ubiquitous, their security vulnerabilities are critical to understand. Recent work shows that text optimizers can produce jailbreaking prompts that bypass moderation and alignment. Drawing from the rich body of work on adversarial machine learning, we approach these attacks with three questions: What threat models are practically useful in this domain? How do baseline defense techniques perform in this new domain? How does LLM security differ from computer vision?   We evaluate several baseline defense strategies against leading adversarial attacks on LLMs, discussing the various settings in which each is feasible and effective. Particularly, we look at three types of defenses: detection (perplexity based), input preprocessing (paraphrase and retokenization), and adversarial training. We discuss white-box and gray-box settings and discuss the robustness-performance trade-off for each of the defenses considered. Surprisingly, we find much more success with filtering and preprocessing than we would expect from other domains, such as vision, providing a first indication that the relative strengths of these defenses may be weighed differently in these domains.



## **2. Why do universal adversarial attacks work on large language models?: Geometry might be the answer**

cs.LG

2nd AdvML Frontiers Workshop at 40th International Conference on  Machine Learning, Honolulu, Hawaii, USA, 2023

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2309.00254v1) [paper-pdf](http://arxiv.org/pdf/2309.00254v1)

**Authors**: Varshini Subhash, Anna Bialas, Weiwei Pan, Finale Doshi-Velez

**Abstract**: Transformer based large language models with emergent capabilities are becoming increasingly ubiquitous in society. However, the task of understanding and interpreting their internal workings, in the context of adversarial attacks, remains largely unsolved. Gradient-based universal adversarial attacks have been shown to be highly effective on large language models and potentially dangerous due to their input-agnostic nature. This work presents a novel geometric perspective explaining universal adversarial attacks on large language models. By attacking the 117M parameter GPT-2 model, we find evidence indicating that universal adversarial triggers could be embedding vectors which merely approximate the semantic information in their adversarial training region. This hypothesis is supported by white-box model analysis comprising dimensionality reduction and similarity measurement of hidden representations. We believe this new geometric perspective on the underlying mechanism driving universal attacks could help us gain deeper insight into the internal workings and failure modes of LLMs, thus enabling their mitigation.



## **3. Image Hijacking: Adversarial Images can Control Generative Models at Runtime**

cs.LG

Code is available at https://github.com/euanong/image-hijacks

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2309.00236v1) [paper-pdf](http://arxiv.org/pdf/2309.00236v1)

**Authors**: Luke Bailey, Euan Ong, Stuart Russell, Scott Emmons

**Abstract**: Are foundation models secure from malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control generative models at runtime. We introduce Behavior Matching, a general method for creating image hijacks, and we use it to explore three types of attacks. Specific string attacks generate arbitrary output of the adversary's choosing. Leak context attacks leak information from the context window into the output. Jailbreak attacks circumvent a model's safety training. We study these attacks against LLaVA-2, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all our attack types have above a 90\% success rate. Moreover, our attacks are automated and require only small image perturbations. These findings raise serious concerns about the security of foundation models. If image hijacks are as difficult to defend against as adversarial examples in CIFAR-10, then it might be many years before a solution is found -- if it even exists.



## **4. Dynamical systems' based neural networks**

cs.LG

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2210.02373v2) [paper-pdf](http://arxiv.org/pdf/2210.02373v2)

**Authors**: Elena Celledoni, Davide Murari, Brynjulf Owren, Carola-Bibiane Schönlieb, Ferdia Sherry

**Abstract**: Neural networks have gained much interest because of their effectiveness in many applications. However, their mathematical properties are generally not well understood. If there is some underlying geometric structure inherent to the data or to the function to approximate, it is often desirable to take this into account in the design of the neural network. In this work, we start with a non-autonomous ODE and build neural networks using a suitable, structure-preserving, numerical time-discretisation. The structure of the neural network is then inferred from the properties of the ODE vector field. Besides injecting more structure into the network architectures, this modelling procedure allows a better theoretical understanding of their behaviour. We present two universal approximation results and demonstrate how to impose some particular properties on the neural networks. A particular focus is on 1-Lipschitz architectures including layers that are not 1-Lipschitz. These networks are expressive and robust against adversarial attacks, as shown for the CIFAR-10 and CIFAR-100 datasets.



## **5. Fault Injection and Safe-Error Attack for Extraction of Embedded Neural Network Models**

cs.CR

Accepted at SECAI Workshop, ESORICS 2023

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16703v1) [paper-pdf](http://arxiv.org/pdf/2308.16703v1)

**Authors**: Kevin Hector, Pierre-Alain Moellic, Mathieu Dumont, Jean-Max Dutertre

**Abstract**: Model extraction emerges as a critical security threat with attack vectors exploiting both algorithmic and implementation-based approaches. The main goal of an attacker is to steal as much information as possible about a protected victim model, so that he can mimic it with a substitute model, even with a limited access to similar training data. Recently, physical attacks such as fault injection have shown worrying efficiency against the integrity and confidentiality of embedded models. We focus on embedded deep neural network models on 32-bit microcontrollers, a widespread family of hardware platforms in IoT, and the use of a standard fault injection strategy - Safe Error Attack (SEA) - to perform a model extraction attack with an adversary having a limited access to training data. Since the attack strongly depends on the input queries, we propose a black-box approach to craft a successful attack set. For a classical convolutional neural network, we successfully recover at least 90% of the most significant bits with about 1500 crafted inputs. These information enable to efficiently train a substitute model, with only 8% of the training dataset, that reaches high fidelity and near identical accuracy level than the victim model.



## **6. Everyone Can Attack: Repurpose Lossy Compression as a Natural Backdoor Attack**

cs.CR

14 pages. This paper shows everyone can mount a powerful and stealthy  backdoor attack with the widely-used lossy image compression

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16684v1) [paper-pdf](http://arxiv.org/pdf/2308.16684v1)

**Authors**: Sze Jue Yang, Quang Nguyen, Chee Seng Chan, Khoa Doan

**Abstract**: The vulnerabilities to backdoor attacks have recently threatened the trustworthiness of machine learning models in practical applications. Conventional wisdom suggests that not everyone can be an attacker since the process of designing the trigger generation algorithm often involves significant effort and extensive experimentation to ensure the attack's stealthiness and effectiveness. Alternatively, this paper shows that there exists a more severe backdoor threat: anyone can exploit an easily-accessible algorithm for silent backdoor attacks. Specifically, this attacker can employ the widely-used lossy image compression from a plethora of compression tools to effortlessly inject a trigger pattern into an image without leaving any noticeable trace; i.e., the generated triggers are natural artifacts. One does not require extensive knowledge to click on the "convert" or "save as" button while using tools for lossy image compression. Via this attack, the adversary does not need to design a trigger generator as seen in prior works and only requires poisoning the data. Empirically, the proposed attack consistently achieves 100% attack success rate in several benchmark datasets such as MNIST, CIFAR-10, GTSRB and CelebA. More significantly, the proposed attack can still achieve almost 100% attack success rate with very small (approximately 10%) poisoning rates in the clean label setting. The generated trigger of the proposed attack using one lossy compression algorithm is also transferable across other related compression algorithms, exacerbating the severity of this backdoor threat. This work takes another crucial step toward understanding the extensive risks of backdoor attacks in practice, urging practitioners to investigate similar attacks and relevant backdoor mitigation methods.



## **7. Fault Injection on Embedded Neural Networks: Impact of a Single Instruction Skip**

cs.CR

Accepted at DSD 2023 for AHSA Special Session

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16665v1) [paper-pdf](http://arxiv.org/pdf/2308.16665v1)

**Authors**: Clement Gaine, Pierre-Alain Moellic, Olivier Potin, Jean-Max Dutertre

**Abstract**: With the large-scale integration and use of neural network models, especially in critical embedded systems, their security assessment to guarantee their reliability is becoming an urgent need. More particularly, models deployed in embedded platforms, such as 32-bit microcontrollers, are physically accessible by adversaries and therefore vulnerable to hardware disturbances. We present the first set of experiments on the use of two fault injection means, electromagnetic and laser injections, applied on neural networks models embedded on a Cortex M4 32-bit microcontroller platform. Contrary to most of state-of-the-art works dedicated to the alteration of the internal parameters or input values, our goal is to simulate and experimentally demonstrate the impact of a specific fault model that is instruction skip. For that purpose, we assessed several modification attacks on the control flow of a neural network inference. We reveal integrity threats by targeting several steps in the inference program of typical convolutional neural network models, which may be exploited by an attacker to alter the predictions of the target models with different adversarial goals.



## **8. Security Allocation in Networked Control Systems under Stealthy Attacks**

eess.SY

11 pages, 3 figures, and 1 table, journal submission

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16639v1) [paper-pdf](http://arxiv.org/pdf/2308.16639v1)

**Authors**: Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper considers the problem of security allocation in a networked control system under stealthy attacks in which the system is comprised of interconnected subsystems represented by vertices. A malicious adversary selects a single vertex on which to conduct a stealthy data injection attack to maximally disrupt the local performance while remaining undetected. On the other hand, a defender selects several vertices on which to allocate defense resources against the adversary. First, the objectives of the adversary and the defender with uncertain targets are formulated in probabilistic ways, resulting in an expected worst-case impact of stealthy attacks. Next, we provide a graph-theoretic necessary and sufficient condition under which the cost for the defender and the expected worst-case impact of stealthy attacks are bounded. This condition enables the defender to restrict the admissible actions to a subset of available vertex sets. Then, we cast the problem of security allocation in a Stackelberg game-theoretic framework. Finally, the contribution of this paper is highlighted by utilizing the proposed admissible actions of the defender in the context of large-scale networks. A numerical example of a 50-vertex networked control system is presented to validate the obtained results.



## **9. The Power of MEME: Adversarial Malware Creation with Model-Based Reinforcement Learning**

cs.CR

12 pages, 3 figures, 3 tables. Accepted at ESORICS 2023

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2308.16562v1) [paper-pdf](http://arxiv.org/pdf/2308.16562v1)

**Authors**: Maria Rigaki, Sebastian Garcia

**Abstract**: Due to the proliferation of malware, defenders are increasingly turning to automation and machine learning as part of the malware detection tool-chain. However, machine learning models are susceptible to adversarial attacks, requiring the testing of model and product robustness. Meanwhile, attackers also seek to automate malware generation and evasion of antivirus systems, and defenders try to gain insight into their methods. This work proposes a new algorithm that combines Malware Evasion and Model Extraction (MEME) attacks. MEME uses model-based reinforcement learning to adversarially modify Windows executable binary samples while simultaneously training a surrogate model with a high agreement with the target model to evade. To evaluate this method, we compare it with two state-of-the-art attacks in adversarial malware creation, using three well-known published models and one antivirus product as targets. Results show that MEME outperforms the state-of-the-art methods in terms of evasion capabilities in almost all cases, producing evasive malware with an evasion rate in the range of 32-73%. It also produces surrogate models with a prediction label agreement with the respective target models between 97-99%. The surrogate could be used to fine-tune and improve the evasion rate in the future.



## **10. Why Does Little Robustness Help? Understanding and Improving Adversarial Transferability from Surrogate Training**

cs.LG

IEEE Symposium on Security and Privacy (Oakland) 2024; Extended  version of camera-ready

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2307.07873v5) [paper-pdf](http://arxiv.org/pdf/2307.07873v5)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.



## **11. Interpretable and Robust AI in EEG Systems: A Survey**

eess.SP

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2304.10755v2) [paper-pdf](http://arxiv.org/pdf/2304.10755v2)

**Authors**: Xinliang Zhou, Chenyu Liu, Liming Zhai, Ziyu Jia, Cuntai Guan, Yang Liu

**Abstract**: The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI's reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it into three types: backpropagation, perturbation, and inherently interpretable methods. Then we classify the robustness mechanisms into four classes: noise and artifacts, human variability, data acquisition instability, and adversarial attacks. Finally, we identify several critical and unresolved challenges for interpretable and robust AI in EEG systems and further discuss their future directions.



## **12. Pre-trained transformer for adversarial purification**

cs.CR

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2306.01762v2) [paper-pdf](http://arxiv.org/pdf/2306.01762v2)

**Authors**: Kai Wu, Yujian Betterest Li, Xiaoyu Zhang, Handing Wang, Jing Liu

**Abstract**: With more and more deep neural networks being deployed as various daily services, their reliability is essential. It's frightening that deep neural networks are vulnerable and sensitive to adversarial attacks, the most common one of which for the services is evasion-based. Recent works usually strengthen the robustness by adversarial training or leveraging the knowledge of an amount of clean data. However, in practical terms, retraining and redeploying the model need a large computational budget, leading to heavy losses to the online service. In addition, when adversarial examples of a certain attack are detected, only limited adversarial examples are available for the service provider, while much clean data may not be accessible. Given the mentioned problems, we propose a new scenario, RaPiD (Rapid Plug-in Defender), which is to rapidly defend against a certain attack for the frozen original service model with limitations of few clean and adversarial examples. Motivated by the generalization and the universal computation ability of pre-trained transformer models, we come up with a new defender method, CeTaD, which stands for Considering Pre-trained Transformers as Defenders. In particular, we evaluate the effectiveness and the transferability of CeTaD in the case of one-shot adversarial examples and explore the impact of different parts of CeTaD as well as training data conditions. CeTaD is flexible, able to be embedded into an arbitrary differentiable model, and suitable for various types of attacks.



## **13. Vulnerability of Machine Learning Approaches Applied in IoT-based Smart Grid: A Review**

cs.CR

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2308.15736v1) [paper-pdf](http://arxiv.org/pdf/2308.15736v1)

**Authors**: Zhenyong Zhang, Mengxiang Liu, Mingyang Sun, Ruilong Deng, Peng Cheng, Dusit Niyato, Mo-Yuen Chow, Jiming Chen

**Abstract**: The machine learning (ML) sees an increasing prevalence of being used in the internet-of-things enabled smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. The survey is organized from the aspects of adversarial assumptions, targeted applications, evaluation metrics, defending approaches, physics-related constraints, and applied datasets. We also highlight future directions on this topic to encourage more researchers to conduct further research on adversarial attacks and defending approaches for MLsgAPPs.



## **14. Intriguing Properties of Diffusion Models: A Large-Scale Dataset for Evaluating Natural Attack Capability in Text-to-Image Generative Models**

cs.CV

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2308.15692v1) [paper-pdf](http://arxiv.org/pdf/2308.15692v1)

**Authors**: Takami Sato, Justin Yue, Nanze Chen, Ningfei Wang, Qi Alfred Chen

**Abstract**: Denoising probabilistic diffusion models have shown breakthrough performance that can generate more photo-realistic images or human-level illustrations than the prior models such as GANs. This high image-generation capability has stimulated the creation of many downstream applications in various areas. However, we find that this technology is indeed a double-edged sword: We identify a new type of attack, called the Natural Denoising Diffusion (NDD) attack based on the finding that state-of-the-art deep neural network (DNN) models still hold their prediction even if we intentionally remove their robust features, which are essential to the human visual system (HVS), by text prompts. The NDD attack can generate low-cost, model-agnostic, and transferrable adversarial attacks by exploiting the natural attack capability in diffusion models. Motivated by the finding, we construct a large-scale dataset, Natural Denoising Diffusion Attack (NDDA) dataset, to systematically evaluate the risk of the natural attack capability of diffusion models with state-of-the-art text-to-image diffusion models. We evaluate the natural attack capability by answering 6 research questions. Through a user study to confirm the validity of the NDD attack, we find that the NDD attack can achieve an 88% detection rate while being stealthy to 93% of human subjects. We also find that the non-robust features embedded by diffusion models contribute to the natural attack capability. To confirm the model-agnostic and transferrable attack capability, we perform the NDD attack against an AD vehicle and find that 73% of the physically printed attacks can be detected as a stop sign. We hope that our study and dataset can help our community to be aware of the risk of diffusion models and facilitate further research toward robust DNN models.



## **15. MDTD: A Multi Domain Trojan Detector for Deep Neural Networks**

cs.CR

Accepted to ACM Conference on Computer and Communications Security  (ACM CCS) 2023

**SubmitDate**: 2023-08-30    [abs](http://arxiv.org/abs/2308.15673v1) [paper-pdf](http://arxiv.org/pdf/2308.15673v1)

**Authors**: Arezoo Rajabi, Surudhi Asokraj, Fengqing Jiang, Luyao Niu, Bhaskar Ramasubramanian, Jim Ritcey, Radha Poovendran

**Abstract**: Machine learning models that use deep neural networks (DNNs) are vulnerable to backdoor attacks. An adversary carrying out a backdoor attack embeds a predefined perturbation called a trigger into a small subset of input samples and trains the DNN such that the presence of the trigger in the input results in an adversary-desired output class. Such adversarial retraining however needs to ensure that outputs for inputs without the trigger remain unaffected and provide high classification accuracy on clean samples. In this paper, we propose MDTD, a Multi-Domain Trojan Detector for DNNs, which detects inputs containing a Trojan trigger at testing time. MDTD does not require knowledge of trigger-embedding strategy of the attacker and can be applied to a pre-trained DNN model with image, audio, or graph-based inputs. MDTD leverages an insight that input samples containing a Trojan trigger are located relatively farther away from a decision boundary than clean samples. MDTD estimates the distance to a decision boundary using adversarial learning methods and uses this distance to infer whether a test-time input sample is Trojaned or not. We evaluate MDTD against state-of-the-art Trojan detection methods across five widely used image-based datasets: CIFAR100, CIFAR10, GTSRB, SVHN, and Flowers102; four graph-based datasets: AIDS, WinMal, Toxicant, and COLLAB; and the SpeechCommand audio dataset. MDTD effectively identifies samples that contain different types of Trojan triggers. We evaluate MDTD against adaptive attacks where an adversary trains a robust DNN to increase (decrease) distance of benign (Trojan) inputs from a decision boundary.



## **16. Adaptive Attack Detection in Text Classification: Leveraging Space Exploration Features for Text Sentiment Classification**

cs.CR

Presented at 2nd International Workshop on Adaptive Cyber Defense,  2023 (arXiv:2308.09520)

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15663v1) [paper-pdf](http://arxiv.org/pdf/2308.15663v1)

**Authors**: Atefeh Mahdavi, Neda Keivandarian, Marco Carvalho

**Abstract**: Adversarial example detection plays a vital role in adaptive cyber defense, especially in the face of rapidly evolving attacks. In adaptive cyber defense, the nature and characteristics of attacks continuously change, making it crucial to have robust mechanisms in place to detect and counter these threats effectively. By incorporating adversarial example detection techniques, adaptive cyber defense systems can enhance their ability to identify and mitigate attacks that attempt to exploit vulnerabilities in machine learning models or other systems. Adversarial examples are inputs that are crafted by applying intentional perturbations to natural inputs that result in incorrect classification. In this paper, we propose a novel approach that leverages the power of BERT (Bidirectional Encoder Representations from Transformers) and introduces the concept of Space Exploration Features. We utilize the feature vectors obtained from the BERT model's output to capture a new representation of feature space to improve the density estimation method.



## **17. Everything Perturbed All at Once: Enabling Differentiable Graph Attacks**

cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15614v1) [paper-pdf](http://arxiv.org/pdf/2308.15614v1)

**Authors**: Haoran Liu, Bokun Wang, Jianling Wang, Xiangjue Dong, Tianbao Yang, James Caverlee

**Abstract**: As powerful tools for representation learning on graphs, graph neural networks (GNNs) have played an important role in applications including social networks, recommendation systems, and online web services. However, GNNs have been shown to be vulnerable to adversarial attacks, which can significantly degrade their effectiveness. Recent state-of-the-art approaches in adversarial attacks rely on gradient-based meta-learning to selectively perturb a single edge with the highest attack score until they reach the budget constraint. While effective in identifying vulnerable links, these methods are plagued by high computational costs. By leveraging continuous relaxation and parameterization of the graph structure, we propose a novel attack method called Differentiable Graph Attack (DGA) to efficiently generate effective attacks and meanwhile eliminate the need for costly retraining. Compared to the state-of-the-art, DGA achieves nearly equivalent attack performance with 6 times less training time and 11 times smaller GPU memory footprint on different benchmark datasets. Additionally, we provide extensive experimental analyses of the transferability of the DGA among different graph models, as well as its robustness against widely-used defense mechanisms.



## **18. Masquerade: Simple and Lightweight Transaction Reordering Mitigation in Blockchains**

cs.CR

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15347v1) [paper-pdf](http://arxiv.org/pdf/2308.15347v1)

**Authors**: Arti Vedula, Shaileshh Bojja Venkatakrishnan, Abhishek Gupta

**Abstract**: Blockchains offer strong security gurarantees, but cannot protect users against the ordering of transactions. Players such as miners, bots and validators can reorder various transactions and reap significant profits, called the Maximal Extractable Value (MEV). In this paper, we propose an MEV aware protocol design called Masquerade, and show that it will increase user satisfaction and confidence in the system. We propose a strict per-transaction level of ordering to ensure that a transaction is committed either way even if it is revealed. In this protocol, we introduce the notion of a "token" to mitigate the actions taken by an adversary in an attack scenario. Such tokens can be purchased voluntarily by users, who can then choose to include the token numbers in their transactions. If the users include the token in their transactions, then our protocol requires the block-builder to order the transactions strictly according to token numbers. We show through extensive simulations that this reduces the probability that the adversaries can benefit from MEV transactions as compared to existing current practices.



## **19. Imperceptible Adversarial Attack on Deep Neural Networks from Image Boundary**

cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15344v1) [paper-pdf](http://arxiv.org/pdf/2308.15344v1)

**Authors**: Fahad Alrasheedi, Xin Zhong

**Abstract**: Although Deep Neural Networks (DNNs), such as the convolutional neural networks (CNN) and Vision Transformers (ViTs), have been successfully applied in the field of computer vision, they are demonstrated to be vulnerable to well-sought Adversarial Examples (AEs) that can easily fool the DNNs. The research in AEs has been active, and many adversarial attacks and explanations have been proposed since they were discovered in 2014. The mystery of the AE's existence is still an open question, and many studies suggest that DNN training algorithms have blind spots. The salient objects usually do not overlap with boundaries; hence, the boundaries are not the DNN model's attention. Nevertheless, recent studies show that the boundaries can dominate the behavior of the DNN models. Hence, this study aims to look at the AEs from a different perspective and proposes an imperceptible adversarial attack that systemically attacks the input image boundary for finding the AEs. The experimental results have shown that the proposed boundary attacking method effectively attacks six CNN models and the ViT using only 32% of the input image content (from the boundaries) with an average success rate (SR) of 95.2% and an average peak signal-to-noise ratio of 41.37 dB. Correlation analyses are conducted, including the relation between the adversarial boundary's width and the SR and how the adversarial boundary changes the DNN model's attention. This paper's discoveries can potentially advance the understanding of AEs and provide a different perspective on how AEs can be constructed.



## **20. Longest-chain Attacks: Difficulty Adjustment and Timestamp Verifiability**

cs.CR

A short version appears at MobiHoc23 as a poster

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15312v1) [paper-pdf](http://arxiv.org/pdf/2308.15312v1)

**Authors**: Tzuo Hann Law, Selman Erol, Lewis Tseng

**Abstract**: We study an adversary who attacks a Proof-of-Work (POW) blockchain by selfishly constructing an alternative longest chain. We characterize optimal strategies employed by the adversary when a difficulty adjustment rule al\`a Bitcoin applies. As time (namely the times-tamp specified in each block) in most permissionless POW blockchains is somewhat subjective, we focus on two extreme scenarios: when time is completely verifiable, and when it is completely unverifiable. We conclude that an adversary who faces a difficulty adjustment rule will find a longest-chain attack very challenging when timestamps are verifiable. POW blockchains with frequent difficulty adjustments relative to time reporting flexibility will be substantially more vulnerable to longest-chain attacks. Our main fining provides guidance on the design of difficulty adjustment rules and demonstrates the importance of timestamp verifiability.



## **21. A Classification-Guided Approach for Adversarial Attacks against Neural Machine Translation**

cs.CL

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15246v1) [paper-pdf](http://arxiv.org/pdf/2308.15246v1)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Neural Machine Translation (NMT) models have been shown to be vulnerable to adversarial attacks, wherein carefully crafted perturbations of the input can mislead the target model. In this paper, we introduce ACT, a novel adversarial attack framework against NMT systems guided by a classifier. In our attack, the adversary aims to craft meaning-preserving adversarial examples whose translations by the NMT model belong to a different class than the original translations in the target language. Unlike previous attacks, our new approach has a more substantial effect on the translation by altering the overall meaning, which leads to a different class determined by a classifier. To evaluate the robustness of NMT models to this attack, we propose enhancements to existing black-box word-replacement-based attacks by incorporating output translations of the target NMT model and the output logits of a classifier within the attack process. Extensive experiments in various settings, including a comparison with existing untargeted attacks, demonstrate that the proposed attack is considerably more successful in altering the class of the output translation and has more effect on the translation. This new paradigm can show the vulnerabilities of NMT systems by focusing on the class of translation rather than the mere translation quality as studied traditionally.



## **22. On the Steganographic Capacity of Selected Learning Models**

cs.LG

arXiv admin note: text overlap with arXiv:2306.17189

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15502v1) [paper-pdf](http://arxiv.org/pdf/2308.15502v1)

**Authors**: Rishit Agrawal, Kelvin Jou, Tanush Obili, Daksh Parikh, Samarth Prajapati, Yash Seth, Charan Sridhar, Nathan Zhang, Mark Stamp

**Abstract**: Machine learning and deep learning models are potential vectors for various attack scenarios. For example, previous research has shown that malware can be hidden in deep learning models. Hiding information in a learning model can be viewed as a form of steganography. In this research, we consider the general question of the steganographic capacity of learning models. Specifically, for a wide range of models, we determine the number of low-order bits of the trained parameters that can be overwritten, without adversely affecting model performance. For each model considered, we graph the accuracy as a function of the number of low-order bits that have been overwritten, and for selected models, we also analyze the steganographic capacity of individual layers. The models that we test include the classic machine learning techniques of Linear Regression (LR) and Support Vector Machine (SVM); the popular general deep learning models of Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN); the highly-successful Recurrent Neural Network (RNN) architecture of Long Short-Term Memory (LSTM); the pre-trained transfer learning-based models VGG16, DenseNet121, InceptionV3, and Xception; and, finally, an Auxiliary Classifier Generative Adversarial Network (ACGAN). In all cases, we find that a majority of the bits of each trained parameter can be overwritten before the accuracy degrades. Of the models tested, the steganographic capacity ranges from 7.04 KB for our LR experiments, to 44.74 MB for InceptionV3. We discuss the implications of our results and consider possible avenues for further research.



## **23. Can We Rely on AI?**

math.NA

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15092v1) [paper-pdf](http://arxiv.org/pdf/2308.15092v1)

**Authors**: Desmond J. Higham

**Abstract**: Over the last decade, adversarial attack algorithms have revealed instabilities in deep learning tools. These algorithms raise issues regarding safety, reliability and interpretability in artificial intelligence; especially in high risk settings. From a practical perspective, there has been a war of escalation between those developing attack and defence strategies. At a more theoretical level, researchers have also studied bigger picture questions concerning the existence and computability of attacks. Here we give a brief overview of the topic, focusing on aspects that are likely to be of interest to researchers in applied and computational mathematics.



## **24. Advancing Adversarial Robustness Through Adversarial Logit Update**

cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15072v1) [paper-pdf](http://arxiv.org/pdf/2308.15072v1)

**Authors**: Hao Xuan, Peican Zhu, Xingyu Li

**Abstract**: Deep Neural Networks are susceptible to adversarial perturbations. Adversarial training and adversarial purification are among the most widely recognized defense strategies. Although these methods have different underlying logic, both rely on absolute logit values to generate label predictions. In this study, we theoretically analyze the logit difference around successful adversarial attacks from a theoretical point of view and propose a new principle, namely Adversarial Logit Update (ALU), to infer adversarial sample's labels. Based on ALU, we introduce a new classification paradigm that utilizes pre- and post-purification logit differences for model's adversarial robustness boost. Without requiring adversarial or additional data for model training, our clean data synthesis model can be easily applied to various pre-trained models for both adversarial sample detection and ALU-based data classification. Extensive experiments on both CIFAR-10, CIFAR-100, and tiny-ImageNet datasets show that even with simple components, the proposed solution achieves superior robustness performance compared to state-of-the-art methods against a wide range of adversarial attacks. Our python implementation is submitted in our Supplementary document and will be published upon the paper's acceptance.



## **25. Double Public Key Signing Function Oracle Attack on EdDSA Software Implementations**

cs.CR

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15009v1) [paper-pdf](http://arxiv.org/pdf/2308.15009v1)

**Authors**: Sam Grierson, Konstantinos Chalkias, William J Buchanan

**Abstract**: EdDSA is a standardised elliptic curve digital signature scheme introduced to overcome some of the issues prevalent in the more established ECDSA standard. Due to the EdDSA standard specifying that the EdDSA signature be deterministic, if the signing function were to be used as a public key signing oracle for the attacker, the unforgeability notion of security of the scheme can be broken. This paper describes an attack against some of the most popular EdDSA implementations, which results in an adversary recovering the private key used during signing. With this recovered secret key, an adversary can sign arbitrary messages that would be seen as valid by the EdDSA verification function. A list of libraries with vulnerable APIs at the time of publication is provided. Furthermore, this paper provides two suggestions for securing EdDSA signing APIs against this vulnerability while it additionally discusses failed attempts to solve the issue.



## **26. Stealthy Backdoor Attack for Code Models**

cs.CR

18 pages, Under review of IEEE Transactions on Software Engineering

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2301.02496v2) [paper-pdf](http://arxiv.org/pdf/2301.02496v2)

**Authors**: Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo

**Abstract**: Code models, such as CodeBERT and CodeT5, offer general-purpose representations of code and play a vital role in supporting downstream automated software engineering tasks. Most recently, code models were revealed to be vulnerable to backdoor attacks. A code model that is backdoor-attacked can behave normally on clean examples but will produce pre-defined malicious outputs on examples injected with triggers that activate the backdoors. Existing backdoor attacks on code models use unstealthy and easy-to-detect triggers. This paper aims to investigate the vulnerability of code models with stealthy backdoor attacks. To this end, we propose AFRAIDOOR (Adversarial Feature as Adaptive Backdoor). AFRAIDOOR achieves stealthiness by leveraging adversarial perturbations to inject adaptive triggers into different inputs. We evaluate AFRAIDOOR on three widely adopted code models (CodeBERT, PLBART and CodeT5) and two downstream tasks (code summarization and method name prediction). We find that around 85% of adaptive triggers in AFRAIDOOR bypass the detection in the defense process. By contrast, only less than 12% of the triggers from previous work bypass the defense. When the defense method is not applied, both AFRAIDOOR and baselines have almost perfect attack success rates. However, once a defense is applied, the success rates of baselines decrease dramatically to 10.47% and 12.06%, while the success rate of AFRAIDOOR are 77.05% and 92.98% on the two tasks. Our finding exposes security weaknesses in code models under stealthy backdoor attacks and shows that the state-of-the-art defense method cannot provide sufficient protection. We call for more research efforts in understanding security threats to code models and developing more effective countermeasures.



## **27. WSAM: Visual Explanations from Style Augmentation as Adversarial Attacker and Their Influence in Image Classification**

cs.CV

8 pages, 10 figures

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.14995v1) [paper-pdf](http://arxiv.org/pdf/2308.14995v1)

**Authors**: Felipe Moreno-Vera, Edgar Medina, Jorge Poco

**Abstract**: Currently, style augmentation is capturing attention due to convolutional neural networks (CNN) being strongly biased toward recognizing textures rather than shapes. Most existing styling methods either perform a low-fidelity style transfer or a weak style representation in the embedding vector. This paper outlines a style augmentation algorithm using stochastic-based sampling with noise addition to improving randomization on a general linear transformation for style transfer. With our augmentation strategy, all models not only present incredible robustness against image stylizing but also outperform all previous methods and surpass the state-of-the-art performance for the STL-10 dataset. In addition, we present an analysis of the model interpretations under different style variations. At the same time, we compare comprehensive experiments demonstrating the performance when applied to deep neural architectures in training settings.



## **28. Randomized Line-to-Row Mapping for Low-Overhead Rowhammer Mitigations**

cs.CR

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14907v1) [paper-pdf](http://arxiv.org/pdf/2308.14907v1)

**Authors**: Anish Saxena, Saurav Mathur, Moinuddin Qureshi

**Abstract**: Modern systems mitigate Rowhammer using victim refresh, which refreshes the two neighbours of an aggressor row when it encounters a specified number of activations. Unfortunately, complex attack patterns like Half-Double break victim-refresh, rendering current systems vulnerable. Instead, recently proposed secure Rowhammer mitigations rely on performing mitigative action on the aggressor rather than the victims. Such schemes employ mitigative actions such as row-migration or access-control and include AQUA, SRS, and Blockhammer. While these schemes incur only modest slowdowns at Rowhammer thresholds of few thousand, they incur prohibitive slowdowns (15%-600%) for lower thresholds that are likely in the near future. The goal of our paper is to make secure Rowhammer mitigations practical at such low thresholds.   Our paper provides the key insights that benign application encounter thousands of hot rows (receiving more activations than the threshold) due to the memory mapping, which places spatially proximate lines in the same row to maximize row-buffer hitrate. Unfortunately, this causes row to receive activations for many frequently used lines. We propose Rubix, which breaks the spatial correlation in the line-to-row mapping by using an encrypted address to access the memory, reducing the likelihood of hot rows by 2 to 3 orders of magnitude. To aid row-buffer hits, Rubix randomizes a group of 1-4 lines. We also propose Rubix-D, which dynamically changes the line-to-row mapping. Rubix-D minimizes hot-rows and makes it much harder for an adversary to learn the spatial neighbourhood of a row. Rubix reduces the slowdown of AQUA (from 15% to 1%), SRS (from 60% to 2%), and Blockhammer (from 600% to 3%) while incurring a storage of less than 1 Kilobyte.



## **29. A Stochastic Surveillance Stackelberg Game: Co-Optimizing Defense Placement and Patrol Strategy**

eess.SY

8 pages, 1 figure, jointly submitted to the IEEE Control Systems  Letters and the 2024 American Control Conference

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14714v1) [paper-pdf](http://arxiv.org/pdf/2308.14714v1)

**Authors**: Yohan John, Gilberto Diaz-Garcia, Xiaoming Duan, Jason R. Marden, Francesco Bullo

**Abstract**: Stochastic patrol routing is known to be advantageous in adversarial settings; however, the optimal choice of stochastic routing strategy is dependent on a model of the adversary. Duan et al. formulated a Stackelberg game for the worst-case scenario, i.e., a surveillance agent confronted with an omniscient attacker [IEEE TCNS, 8(2), 769-80, 2021]. In this article, we extend their formulation to accommodate heterogeneous defenses at the various nodes of the graph. We derive an upper bound on the value of the game. We identify methods for computing effective patrol strategies for certain classes of graphs. Finally, we leverage the heterogeneous defense formulation to develop novel defense placement algorithms that complement the patrol strategies.



## **30. Adversarial Attacks on Foundational Vision Models**

cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14597v1) [paper-pdf](http://arxiv.org/pdf/2308.14597v1)

**Authors**: Nathan Inkawhich, Gwendolyn McDonald, Ryan Luley

**Abstract**: Rapid progress is being made in developing large, pretrained, task-agnostic foundational vision models such as CLIP, ALIGN, DINOv2, etc. In fact, we are approaching the point where these models do not have to be finetuned downstream, and can simply be used in zero-shot or with a lightweight probing head. Critically, given the complexity of working at this scale, there is a bottleneck where relatively few organizations in the world are executing the training then sharing the models on centralized platforms such as HuggingFace and torch.hub. The goal of this work is to identify several key adversarial vulnerabilities of these models in an effort to make future designs more robust. Intuitively, our attacks manipulate deep feature representations to fool an out-of-distribution (OOD) detector which will be required when using these open-world-aware models to solve closed-set downstream tasks. Our methods reliably make in-distribution (ID) images (w.r.t. a downstream task) be predicted as OOD and vice versa while existing in extremely low-knowledge-assumption threat models. We show our attacks to be potent in whitebox and blackbox settings, as well as when transferred across foundational model types (e.g., attack DINOv2 with CLIP)! This work is only just the beginning of a long journey towards adversarially robust foundational vision models.



## **31. ReMAV: Reward Modeling of Autonomous Vehicles for Finding Likely Failure Events**

cs.AI

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14550v1) [paper-pdf](http://arxiv.org/pdf/2308.14550v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous vehicles are advanced driving systems that are well known for being vulnerable to various adversarial attacks, compromising the vehicle's safety, and posing danger to other road users. Rather than actively training complex adversaries by interacting with the environment, there is a need to first intelligently find and reduce the search space to only those states where autonomous vehicles are found less confident. In this paper, we propose a blackbox testing framework ReMAV using offline trajectories first to analyze the existing behavior of autonomous vehicles and determine appropriate thresholds for finding the probability of failure events. Our reward modeling technique helps in creating a behavior representation that allows us to highlight regions of likely uncertain behavior even when the baseline autonomous vehicle is performing well. This approach allows for more efficient testing without the need for computational and inefficient active adversarial learning techniques. We perform our experiments in a high-fidelity urban driving environment using three different driving scenarios containing single and multi-agent interactions. Our experiment shows 35%, 23%, 48%, and 50% increase in occurrences of vehicle collision, road objects collision, pedestrian collision, and offroad steering events respectively by the autonomous vehicle under test, demonstrating a significant increase in failure events. We also perform a comparative analysis with prior testing frameworks and show that they underperform in terms of training-testing efficiency, finding total infractions, and simulation steps to identify the first failure compared to our approach. The results show that the proposed framework can be used to understand existing weaknesses of the autonomous vehicles under test in order to only attack those regions, starting with the simplistic perturbation models.



## **32. Efficient Decision-based Black-box Patch Attacks on Video Recognition**

cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2303.11917v2) [paper-pdf](http://arxiv.org/pdf/2303.11917v2)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Hao Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, Wenqiang Zhang

**Abstract**: Although Deep Neural Networks (DNNs) have demonstrated excellent performance, they are vulnerable to adversarial patches that introduce perceptible and localized perturbations to the input. Generating adversarial patches on images has received much attention, while adversarial patches on videos have not been well investigated. Further, decision-based attacks, where attackers only access the predicted hard labels by querying threat models, have not been well explored on video models either, even if they are practical in real-world video recognition scenes. The absence of such studies leads to a huge gap in the robustness assessment for video models. To bridge this gap, this work first explores decision-based patch attacks on video models. We analyze that the huge parameter space brought by videos and the minimal information returned by decision-based models both greatly increase the attack difficulty and query burden. To achieve a query-efficient attack, we propose a spatial-temporal differential evolution (STDE) framework. First, STDE introduces target videos as patch textures and only adds patches on keyframes that are adaptively selected by temporal difference. Second, STDE takes minimizing the patch area as the optimization objective and adopts spatialtemporal mutation and crossover to search for the global optimum without falling into the local optimum. Experiments show STDE has demonstrated state-of-the-art performance in terms of threat, efficiency and imperceptibility. Hence, STDE has the potential to be a powerful tool for evaluating the robustness of video recognition models.



## **33. Mitigating the source-side channel vulnerability by characterization of photon statistics**

quant-ph

Comments and suggestions are welcomed

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14402v1) [paper-pdf](http://arxiv.org/pdf/2308.14402v1)

**Authors**: Tanya Sharma, Ayan Biswas, Jayanth Ramakrishnan, Pooja Chandravanshi, Ravindra P. Singh

**Abstract**: Quantum key distribution (QKD) theoretically offers unconditional security. Unfortunately, the gap between theory and practice threatens side-channel attacks on practical QKD systems. Many well-known QKD protocols use weak coherent laser pulses to encode the quantum information. These sources differ from ideal single photon sources and follow Poisson statistics. Many protocols, such as decoy state and coincidence detection protocols, rely on monitoring the photon statistics to detect any information leakage. The accurate measurement and characterization of photon statistics enable the detection of adversarial attacks and the estimation of secure key rates, strengthening the overall security of the QKD system. We have rigorously characterized our source to estimate the mean photon number employing multiple detectors for comparison against measurements made with a single detector. Furthermore, we have also studied intensity fluctuations to help identify and mitigate any potential information leakage due to state preparation flaws. We aim to bridge the gap between theory and practice to achieve information-theoretic security.



## **34. QEVSEC: Quick Electric Vehicle SEcure Charging via Dynamic Wireless Power Transfer**

cs.CR

6 pages, conference

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2205.10292v3) [paper-pdf](http://arxiv.org/pdf/2205.10292v3)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstract**: Dynamic Wireless Power Transfer (DWPT) can be used for on-demand recharging of Electric Vehicles (EV) while driving. However, DWPT raises numerous security and privacy concerns. Recently, researchers demonstrated that DWPT systems are vulnerable to adversarial attacks. In an EV charging scenario, an attacker can prevent the authorized customer from charging, obtain a free charge by billing a victim user and track a target vehicle. State-of-the-art authentication schemes relying on centralized solutions are either vulnerable to various attacks or have high computational complexity, making them unsuitable for a dynamic scenario. In this paper, we propose Quick Electric Vehicle SEcure Charging (QEVSEC), a novel, secure, and efficient authentication protocol for the dynamic charging of EVs. Our idea for QEVSEC originates from multiple vulnerabilities we found in the state-of-the-art protocol that allows tracking of user activity and is susceptible to replay attacks. Based on these observations, the proposed protocol solves these issues and achieves lower computational complexity by using only primitive cryptographic operations in a very short message exchange. QEVSEC provides scalability and a reduced cost in each iteration, thus lowering the impact on the power needed from the grid.



## **35. Hiding Visual Information via Obfuscating Adversarial Perturbations**

cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2209.15304v4) [paper-pdf](http://arxiv.org/pdf/2209.15304v4)

**Authors**: Zhigang Su, Dawei Zhou, Nannan Wangu, Decheng Li, Zhen Wang, Xinbo Gao

**Abstract**: Growing leakage and misuse of visual information raise security and privacy concerns, which promotes the development of information protection. Existing adversarial perturbations-based methods mainly focus on the de-identification against deep learning models. However, the inherent visual information of the data has not been well protected. In this work, inspired by the Type-I adversarial attack, we propose an adversarial visual information hiding method to protect the visual privacy of data. Specifically, the method generates obfuscating adversarial perturbations to obscure the visual information of the data. Meanwhile, it maintains the hidden objectives to be correctly predicted by models. In addition, our method does not modify the parameters of the applied model, which makes it flexible for different scenarios. Experimental results on the recognition and classification tasks demonstrate that the proposed method can effectively hide visual information and hardly affect the performances of models. The code is available in the supplementary material.



## **36. Detecting Language Model Attacks with Perplexity**

cs.CL

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14132v1) [paper-pdf](http://arxiv.org/pdf/2308.14132v1)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, leveraging adversarial suffixes to trick models into generating perilous responses. This method has garnered considerable attention from reputable media outlets such as the New York Times and Wired, thereby influencing public perception regarding the security and safety of LLMs. In this study, we advocate the utilization of perplexity as one of the means to recognize such potential attacks. The underlying concept behind these hacks revolves around appending an unusually constructed string of text to a harmful query that would otherwise be blocked. This maneuver confuses the protective mechanisms and tricks the model into generating a forbidden response. Such scenarios could result in providing detailed instructions to a malicious user for constructing explosives or orchestrating a bank heist. Our investigation demonstrates the feasibility of employing perplexity, a prevalent natural language processing metric, to detect these adversarial tactics before generating a forbidden response. By evaluating the perplexity of queries with and without such adversarial suffixes using an open-source LLM, we discovered that nearly 90 percent were above a perplexity of 1000. This contrast underscores the efficacy of perplexity for detecting this type of exploit.



## **37. Fairness and Privacy in Voice Biometrics:A Study of Gender Influences Using wav2vec 2.0**

eess.AS

7 pages

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14049v1) [paper-pdf](http://arxiv.org/pdf/2308.14049v1)

**Authors**: Oubaida Chouchane, Michele Panariello, Chiara Galdi, Massimiliano Todisco, Nicholas Evans

**Abstract**: This study investigates the impact of gender information on utility, privacy, and fairness in voice biometric systems, guided by the General Data Protection Regulation (GDPR) mandates, which underscore the need for minimizing the processing and storage of private and sensitive data, and ensuring fairness in automated decision-making systems. We adopt an approach that involves the fine-tuning of the wav2vec 2.0 model for speaker verification tasks, evaluating potential gender-related privacy vulnerabilities in the process. Gender influences during the fine-tuning process were employed to enhance fairness and privacy in order to emphasise or obscure gender information within the speakers' embeddings. Results from VoxCeleb datasets indicate our adversarial model increases privacy against uninformed attacks, yet slightly diminishes speaker verification performance compared to the non-adversarial model. However, the model's efficacy reduces against informed attacks. Analysis of system performance was conducted to identify potential gender biases, thus highlighting the need for further research to understand and improve the delicate interplay between utility, privacy, and equity in voice biometric systems.



## **38. Device-Independent Quantum Key Distribution Based on the Mermin-Peres Magic Square Game**

quant-ph

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14037v1) [paper-pdf](http://arxiv.org/pdf/2308.14037v1)

**Authors**: Yi-Zheng Zhen, Yingqiu Mao, Yu-Zhe Zhang, Feihu Xu, Barry C. Sanders

**Abstract**: Device-independent quantum key distribution (DIQKD) is information-theoretically secure against adversaries who possess a scalable quantum computer and who have supplied malicious key-establishment systems; however, the DIQKD key rate is currently too low. Consequently, we devise a DIQKD scheme based on the quantum nonlocal Mermin-Peres magic square game: our scheme asymptotically delivers DIQKD against collective attacks, even with noise. Our scheme outperforms DIQKD using the Clauser-Horne-Shimony-Holt game with respect to the number of game rounds, albeit not number of entangled pairs, provided that both state visibility and detection efficiency are high enough.



## **39. A semantic backdoor attack against Graph Convolutional Networks**

cs.LG

**SubmitDate**: 2023-08-26    [abs](http://arxiv.org/abs/2302.14353v4) [paper-pdf](http://arxiv.org/pdf/2302.14353v4)

**Authors**: Jiazhu Dai, Zhipeng Xiong

**Abstract**: Graph convolutional networks (GCNs) have been very effective in addressing the issue of various graph-structured related tasks. However, recent research has shown that GCNs are vulnerable to a new type of threat called a backdoor attack, where the adversary can inject a hidden backdoor into GCNs so that the attacked model performs well on benign samples, but its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. A semantic backdoor attack is a new type of backdoor attack on deep neural networks (DNNs), where a naturally occurring semantic feature of samples can serve as a backdoor trigger such that the infected DNN models will misclassify testing samples containing the predefined semantic feature even without the requirement of modifying the testing samples. Since the backdoor trigger is a naturally occurring semantic feature of the samples, semantic backdoor attacks are more imperceptible and pose a new and serious threat. In this paper, we investigate whether such semantic backdoor attacks are possible for GCNs and propose a semantic backdoor attack against GCNs (SBAG) under the context of graph classification to reveal the existence of this security vulnerability in GCNs. SBAG uses a certain type of node in the samples as a backdoor trigger and injects a hidden backdoor into GCN models by poisoning training data. The backdoor will be activated, and the GCN models will give malicious classification results specified by the attacker even on unmodified samples as long as the samples contain enough trigger nodes. We evaluate SBAG on four graph datasets and the experimental results indicate that SBAG is effective.



## **40. Active learning for fast and slow modeling attacks on Arbiter PUFs**

cs.CR

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.13645v1) [paper-pdf](http://arxiv.org/pdf/2308.13645v1)

**Authors**: Vincent Dumoulin, Wenjing Rao, Natasha Devroye

**Abstract**: Modeling attacks, in which an adversary uses machine learning techniques to model a hardware-based Physically Unclonable Function (PUF) pose a great threat to the viability of these hardware security primitives. In most modeling attacks, a random subset of challenge-response-pairs (CRPs) are used as the labeled data for the machine learning algorithm. Here, for the arbiter-PUF, a delay based PUF which may be viewed as a linear threshold function with random weights (due to manufacturing imperfections), we investigate the role of active learning in Support Vector Machine (SVM) learning. We focus on challenge selection to help SVM algorithm learn ``fast'' and learn ``slow''. Our methods construct challenges rather than relying on a sample pool of challenges as in prior work. Using active learning to learn ``fast'' (less CRPs revealed, higher accuracies) may help manufacturers learn the manufactured PUFs more efficiently, or may form a more powerful attack when the attacker may query the PUF for CRPs at will. Using active learning to select challenges from which learning is ``slow'' (low accuracy despite a large number of revealed CRPs) may provide a basis for slowing down attackers who are limited to overhearing CRPs.



## **41. Unveiling the Role of Message Passing in Dual-Privacy Preservation on GNNs**

cs.LG

CIKM 2023

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.13513v1) [paper-pdf](http://arxiv.org/pdf/2308.13513v1)

**Authors**: Tianyi Zhao, Hui Hu, Lu Cheng

**Abstract**: Graph Neural Networks (GNNs) are powerful tools for learning representations on graphs, such as social networks. However, their vulnerability to privacy inference attacks restricts their practicality, especially in high-stake domains. To address this issue, privacy-preserving GNNs have been proposed, focusing on preserving node and/or link privacy. This work takes a step back and investigates how GNNs contribute to privacy leakage. Through theoretical analysis and simulations, we identify message passing under structural bias as the core component that allows GNNs to \textit{propagate} and \textit{amplify} privacy leakage. Building upon these findings, we propose a principled privacy-preserving GNN framework that effectively safeguards both node and link privacy, referred to as dual-privacy preservation. The framework comprises three major modules: a Sensitive Information Obfuscation Module that removes sensitive information from node embeddings, a Dynamic Structure Debiasing Module that dynamically corrects the structural bias, and an Adversarial Learning Module that optimizes the privacy-utility trade-off. Experimental results on four benchmark datasets validate the effectiveness of the proposed model in protecting both node and link privacy while preserving high utility for downstream tasks, such as node classification.



## **42. Overcoming Adversarial Attacks for Human-in-the-Loop Applications**

cs.LG

New Frontiers in Adversarial Machine Learning, ICML 2022

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2306.05952v2) [paper-pdf](http://arxiv.org/pdf/2306.05952v2)

**Authors**: Ryan McCoppin, Marla Kennedy, Platon Lukyanenko, Sean Kennedy

**Abstract**: Including human analysis has the potential to positively affect the robustness of Deep Neural Networks and is relatively unexplored in the Adversarial Machine Learning literature. Neural network visual explanation maps have been shown to be prone to adversarial attacks. Further research is needed in order to select robust visualizations of explanations for the image analyst to evaluate a given model. These factors greatly impact Human-In-The-Loop (HITL) evaluation tools due to their reliance on adversarial images, including explanation maps and measurements of robustness. We believe models of human visual attention may improve interpretability and robustness of human-machine imagery analysis systems. Our challenge remains, how can HITL evaluation be robust in this adversarial landscape?



## **43. Defensive Few-shot Learning**

cs.CV

Accepted to IEEE Transactions on Pattern Analysis and Machine  Intelligence (TPAMI) 2022

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/1911.06968v2) [paper-pdf](http://arxiv.org/pdf/1911.06968v2)

**Authors**: Wenbin Li, Lei Wang, Xingxing Zhang, Lei Qi, Jing Huo, Yang Gao, Jiebo Luo

**Abstract**: This paper investigates a new challenging problem called defensive few-shot learning in order to learn a robust few-shot model against adversarial attacks. Simply applying the existing adversarial defense methods to few-shot learning cannot effectively solve this problem. This is because the commonly assumed sample-level distribution consistency between the training and test sets can no longer be met in the few-shot setting. To address this situation, we develop a general defensive few-shot learning (DFSL) framework to answer the following two key questions: (1) how to transfer adversarial defense knowledge from one sample distribution to another? (2) how to narrow the distribution gap between clean and adversarial examples under the few-shot setting? To answer the first question, we propose an episode-based adversarial training mechanism by assuming a task-level distribution consistency to better transfer the adversarial defense knowledge. As for the second question, within each few-shot task, we design two kinds of distribution consistency criteria to narrow the distribution gap between clean and adversarial examples from the feature-wise and prediction-wise perspectives, respectively. Extensive experiments demonstrate that the proposed framework can effectively make the existing few-shot models robust against adversarial attacks. Code is available at https://github.com/WenbinLee/DefensiveFSL.git.



## **44. Feature Unlearning for Pre-trained GANs and VAEs**

cs.CV

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2303.05699v3) [paper-pdf](http://arxiv.org/pdf/2303.05699v3)

**Authors**: Saemi Moon, Seunghyuk Cho, Dongwoo Kim

**Abstract**: We tackle the problem of feature unlearning from a pre-trained image generative model: GANs and VAEs. Unlike a common unlearning task where an unlearning target is a subset of the training set, we aim to unlearn a specific feature, such as hairstyle from facial images, from the pre-trained generative models. As the target feature is only presented in a local region of an image, unlearning the entire image from the pre-trained model may result in losing other details in the remaining region of the image. To specify which features to unlearn, we collect randomly generated images that contain the target features. We then identify a latent representation corresponding to the target feature and then use the representation to fine-tune the pre-trained model. Through experiments on MNIST and CelebA datasets, we show that target features are successfully removed while keeping the fidelity of the original models. Further experiments with an adversarial attack show that the unlearned model is more robust under the presence of malicious parties.



## **45. Face Encryption via Frequency-Restricted Identity-Agnostic Attacks**

cs.CV

I noticed something missing in the article's description in  subsection 3.2, so I'd like to undo it and re-finalize and describe it

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.05983v3) [paper-pdf](http://arxiv.org/pdf/2308.05983v3)

**Authors**: Xin Dong, Rui Wang, Siyuan Liang, Aishan Liu, Lihua Jing

**Abstract**: Billions of people are sharing their daily live images on social media everyday. However, malicious collectors use deep face recognition systems to easily steal their biometric information (e.g., faces) from these images. Some studies are being conducted to generate encrypted face photos using adversarial attacks by introducing imperceptible perturbations to reduce face information leakage. However, existing studies need stronger black-box scenario feasibility and more natural visual appearances, which challenge the feasibility of privacy protection. To address these problems, we propose a frequency-restricted identity-agnostic (FRIA) framework to encrypt face images from unauthorized face recognition without access to personal information. As for the weak black-box scenario feasibility, we obverse that representations of the average feature in multiple face recognition models are similar, thus we propose to utilize the average feature via the crawled dataset from the Internet as the target to guide the generation, which is also agnostic to identities of unknown face recognition systems; in nature, the low-frequency perturbations are more visually perceptible by the human vision system. Inspired by this, we restrict the perturbation in the low-frequency facial regions by discrete cosine transform to achieve the visual naturalness guarantee. Extensive experiments on several face recognition models demonstrate that our FRIA outperforms other state-of-the-art methods in generating more natural encrypted faces while attaining high black-box attack success rates of 96%. In addition, we validate the efficacy of FRIA using real-world black-box commercial API, which reveals the potential of FRIA in practice. Our codes can be found in https://github.com/XinDong10/FRIA.



## **46. Evaluating the Vulnerabilities in ML systems in terms of adversarial attacks**

cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12918v1) [paper-pdf](http://arxiv.org/pdf/2308.12918v1)

**Authors**: John Harshith, Mantej Singh Gill, Madhan Jothimani

**Abstract**: There have been recent adversarial attacks that are difficult to find. These new adversarial attacks methods may pose challenges to current deep learning cyber defense systems and could influence the future defense of cyberattacks. The authors focus on this domain in this research paper. They explore the consequences of vulnerabilities in AI systems. This includes discussing how they might arise, differences between randomized and adversarial examples and also potential ethical implications of vulnerabilities. Moreover, it is important to train the AI systems appropriately when they are in testing phase and getting them ready for broader use.



## **47. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

Appeared at ICML 2023 AdvML Workshop

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2008.09312v6) [paper-pdf](http://arxiv.org/pdf/2008.09312v6)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.



## **48. Fast Adversarial Training with Smooth Convergence**

cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12857v1) [paper-pdf](http://arxiv.org/pdf/2308.12857v1)

**Authors**: Mengnan Zhao, Lihe Zhang, Yuqiu Kong, Baocai Yin

**Abstract**: Fast adversarial training (FAT) is beneficial for improving the adversarial robustness of neural networks. However, previous FAT work has encountered a significant issue known as catastrophic overfitting when dealing with large perturbation budgets, \ie the adversarial robustness of models declines to near zero during training.   To address this, we analyze the training process of prior FAT work and observe that catastrophic overfitting is accompanied by the appearance of loss convergence outliers.   Therefore, we argue a moderately smooth loss convergence process will be a stable FAT process that solves catastrophic overfitting.   To obtain a smooth loss convergence process, we propose a novel oscillatory constraint (dubbed ConvergeSmooth) to limit the loss difference between adjacent epochs. The convergence stride of ConvergeSmooth is introduced to balance convergence and smoothing. Likewise, we design weight centralization without introducing additional hyperparameters other than the loss balance coefficient.   Our proposed methods are attack-agnostic and thus can improve the training stability of various FAT techniques.   Extensive experiments on popular datasets show that the proposed methods efficiently avoid catastrophic overfitting and outperform all previous FAT methods. Code is available at \url{https://github.com/FAT-CS/ConvergeSmooth}.



## **49. Unifying Gradients to Improve Real-world Robustness for Deep Networks**

stat.ML

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2208.06228v2) [paper-pdf](http://arxiv.org/pdf/2208.06228v2)

**Authors**: Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstract**: The wide application of deep neural networks (DNNs) demands an increasing amount of attention to their real-world robustness, i.e., whether a DNN resists black-box adversarial attacks, among which score-based query attacks (SQAs) are most threatening since they can effectively hurt a victim network with the only access to model outputs. Defending against SQAs requires a slight but artful variation of outputs due to the service purpose for users, who share the same output information with SQAs. In this paper, we propose a real-world defense by Unifying Gradients (UniG) of different data so that SQAs could only probe a much weaker attack direction that is similar for different samples. Since such universal attack perturbations have been validated as less aggressive than the input-specific perturbations, UniG protects real-world DNNs by indicating attackers a twisted and less informative attack direction. We implement UniG efficiently by a Hadamard product module which is plug-and-play. According to extensive experiments on 5 SQAs, 2 adaptive attacks and 7 defense baselines, UniG significantly improves real-world robustness without hurting clean accuracy on CIFAR10 and ImageNet. For instance, UniG maintains a model of 77.80% accuracy under 2500-query Square attack while the state-of-the-art adversarially-trained model only has 67.34% on CIFAR10. Simultaneously, UniG outperforms all compared baselines in terms of clean accuracy and achieves the smallest modification of the model output. The code is released at https://github.com/snowien/UniG-pytorch.



## **50. Universal Soldier: Using Universal Adversarial Perturbations for Detecting Backdoor Attacks**

cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2302.00747v3) [paper-pdf](http://arxiv.org/pdf/2302.00747v3)

**Authors**: Xiaoyun Xu, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Deep learning models achieve excellent performance in numerous machine learning tasks. Yet, they suffer from security-related issues such as adversarial examples and poisoning (backdoor) attacks. A deep learning model may be poisoned by training with backdoored data or by modifying inner network parameters. Then, a backdoored model performs as expected when receiving a clean input, but it misclassifies when receiving a backdoored input stamped with a pre-designed pattern called "trigger". Unfortunately, it is difficult to distinguish between clean and backdoored models without prior knowledge of the trigger. This paper proposes a backdoor detection method by utilizing a special type of adversarial attack, universal adversarial perturbation (UAP), and its similarities with a backdoor trigger. We observe an intuitive phenomenon: UAPs generated from backdoored models need fewer perturbations to mislead the model than UAPs from clean models. UAPs of backdoored models tend to exploit the shortcut from all classes to the target class, built by the backdoor trigger. We propose a novel method called Universal Soldier for Backdoor detection (USB) and reverse engineering potential backdoor triggers via UAPs. Experiments on 345 models trained on several datasets show that USB effectively detects the injected backdoor and provides comparable or better results than state-of-the-art methods.



