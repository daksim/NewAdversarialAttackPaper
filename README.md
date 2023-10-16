# Latest Adversarial Attack Papers
**update at 2023-10-16 15:13:22**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. A Stochastic Surveillance Stackelberg Game: Co-Optimizing Defense Placement and Patrol Strategy**

eess.SY

8 pages, 1 figure, jointly submitted to the IEEE Control Systems  Letters and the 2024 American Control Conference. Replaced to fix typos

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2308.14714v2) [paper-pdf](http://arxiv.org/pdf/2308.14714v2)

**Authors**: Yohan John, Gilberto Diaz-Garcia, Xiaoming Duan, Jason R. Marden, Francesco Bullo

**Abstract**: Stochastic patrol routing is known to be advantageous in adversarial settings; however, the optimal choice of stochastic routing strategy is dependent on a model of the adversary. Duan et al. formulated a Stackelberg game for the worst-case scenario, i.e., a surveillance agent confronted with an omniscient attacker [IEEE TCNS, 8(2), 769-80, 2021]. In this article, we extend their formulation to accommodate heterogeneous defenses at the various nodes of the graph. We derive an upper bound on the value of the game. We identify methods for computing effective patrol strategies for certain classes of graphs. Finally, we leverage the heterogeneous defense formulation to develop novel defense placement algorithms that complement the patrol strategies.



## **2. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.03684v2) [paper-pdf](http://arxiv.org/pdf/2310.03684v2)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.



## **3. Worst-Case Morphs using Wasserstein ALI and Improved MIPGAN**

cs.CV

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.08371v2) [paper-pdf](http://arxiv.org/pdf/2310.08371v2)

**Authors**: Una M. Kelly, Meike Nauta, Lu Liu, Luuk J. Spreeuwers, Raymond N. J. Veldhuis

**Abstract**: A morph is a combination of two separate facial images and contains identity information of two different people. When used in an identity document, both people can be authenticated by a biometric Face Recognition (FR) system. Morphs can be generated using either a landmark-based approach or approaches based on deep learning such as Generative Adversarial Networks (GAN). In a recent paper, we introduced a \emph{worst-case} upper bound on how challenging morphing attacks can be for an FR system. The closer morphs are to this upper bound, the bigger the challenge they pose to FR. We introduced an approach with which it was possible to generate morphs that approximate this upper bound for a known FR system (white box), but not for unknown (black box) FR systems.   In this paper, we introduce a morph generation method that can approximate worst-case morphs even when the FR system is not known. A key contribution is that we include the goal of generating difficult morphs \emph{during} training. Our method is based on Adversarially Learned Inference (ALI) and uses concepts from Wasserstein GANs trained with Gradient Penalty, which were introduced to stabilise the training of GANs. We include these concepts to achieve similar improvement in training stability and call the resulting method Wasserstein ALI (WALI). We finetune WALI using loss functions designed specifically to improve the ability to manipulate identity information in facial images and show how it can generate morphs that are more challenging for FR systems than landmark- or GAN-based morphs. We also show how our findings can be used to improve MIPGAN, an existing StyleGAN-based morph generator.



## **4. Attacking The Assortativity Coefficient Under A Rewiring Strategy**

cs.SI

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.08924v1) [paper-pdf](http://arxiv.org/pdf/2310.08924v1)

**Authors**: Shuo Zou, Bo Zhou, Qi Xuan

**Abstract**: Degree correlation is an important characteristic of networks, which is usually quantified by the assortativity coefficient. However, concerns arise about changing the assortativity coefficient of a network when networks suffer from adversarial attacks. In this paper, we analyze the factors that affect the assortativity coefficient and study the optimization problem of maximizing or minimizing the assortativity coefficient (r) in rewired networks with $k$ pairs of edges. We propose a greedy algorithm and formulate the optimization problem using integer programming to obtain the optimal solution for this problem. Through experiments, we demonstrate the reasonableness and effectiveness of our proposed algorithm. For example, rewired edges 10% in the ER network, the assortativity coefficient improved by 60%.



## **5. OTJR: Optimal Transport Meets Optimal Jacobian Regularization for Adversarial Robustness**

cs.CV

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2303.11793v2) [paper-pdf](http://arxiv.org/pdf/2303.11793v2)

**Authors**: Binh M. Le, Shahroz Tariq, Simon S. Woo

**Abstract**: The Web, as a rich medium of diverse content, has been constantly under the threat of malicious entities exploiting its vulnerabilities, especially with the rapid proliferation of deep learning applications in various web services. One such vulnerability, crucial to the fidelity and integrity of web content, is the susceptibility of deep neural networks to adversarial perturbations, especially concerning images - a dominant form of data on the web. In light of the recent advancements in the robustness of classifiers, we delve deep into the intricacies of adversarial training (AT) and Jacobian regularization, two pivotal defenses. Our work {is the} first carefully analyzes and characterizes these two schools of approaches, both theoretically and empirically, to demonstrate how each approach impacts the robust learning of a classifier. Next, we propose our novel Optimal Transport with Jacobian regularization method, dubbed~\SystemName, jointly incorporating the input-output Jacobian regularization into the AT by leveraging the optimal transport theory. In particular, we employ the Sliced Wasserstein (SW) distance that can efficiently push the adversarial samples' representations closer to those of clean samples, regardless of the number of classes within the dataset. The SW distance provides the adversarial samples' movement directions, which are much more informative and powerful for the Jacobian regularization. Our empirical evaluations set a new standard in the domain, with our method achieving commendable accuracies of 51.41\% on the ~\CIFAR-10 and 28.49\% on the ~\CIFAR-100 datasets under the AutoAttack metric. In a real-world demonstration, we subject images sourced from the Internet to online adversarial attacks, reinforcing the efficacy and relevance of our model in defending against sophisticated web-image perturbations.



## **6. Attacks Meet Interpretability (AmI) Evaluation and Findings**

cs.CR

5 pages, 4 figures

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.08808v1) [paper-pdf](http://arxiv.org/pdf/2310.08808v1)

**Authors**: Qian Ma, Ziping Ye, Shagufta Mehnaz

**Abstract**: To investigate the effectiveness of the model explanation in detecting adversarial examples, we reproduce the results of two papers, Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples and Is AmI (Attacks Meet Interpretability) Robust to Adversarial Examples. And then conduct experiments and case studies to identify the limitations of both works. We find that Attacks Meet Interpretability(AmI) is highly dependent on the selection of hyperparameters. Therefore, with a different hyperparameter choice, AmI is still able to detect Nicholas Carlini's attack. Finally, we propose recommendations for future work on the evaluation of defense techniques such as AmI.



## **7. Fed-Safe: Securing Federated Learning in Healthcare Against Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08681v1) [paper-pdf](http://arxiv.org/pdf/2310.08681v1)

**Authors**: Erfan Darzi, Nanna M. Sijtsema, P. M. A van Ooijen

**Abstract**: This paper explores the security aspects of federated learning applications in medical image analysis. Current robustness-oriented methods like adversarial training, secure aggregation, and homomorphic encryption often risk privacy compromises. The central aim is to defend the network against potential privacy breaches while maintaining model robustness against adversarial manipulations. We show that incorporating distributed noise, grounded in the privacy guarantees in federated settings, enables the development of a adversarially robust model that also meets federated privacy standards. We conducted comprehensive evaluations across diverse attack scenarios, parameters, and use cases in cancer imaging, concentrating on pathology, meningioma, and glioma. The results reveal that the incorporation of distributed noise allows for the attainment of security levels comparable to those of conventional adversarial training while requiring fewer retraining samples to establish a robust model.



## **8. Unclonable Non-Interactive Zero-Knowledge**

cs.CR

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.07118v2) [paper-pdf](http://arxiv.org/pdf/2310.07118v2)

**Authors**: Ruta Jawale, Dakshita Khurana

**Abstract**: A non-interactive ZK (NIZK) proof enables verification of NP statements without revealing secrets about them. However, an adversary that obtains a NIZK proof may be able to clone this proof and distribute arbitrarily many copies of it to various entities: this is inevitable for any proof that takes the form of a classical string. In this paper, we ask whether it is possible to rely on quantum information in order to build NIZK proof systems that are impossible to clone.   We define and construct unclonable non-interactive zero-knowledge proofs (of knowledge) for NP. Besides satisfying the zero-knowledge and proof of knowledge properties, these proofs additionally satisfy unclonability. Very roughly, this ensures that no adversary can split an honestly generated proof of membership of an instance $x$ in an NP language $\mathcal{L}$ and distribute copies to multiple entities that all obtain accepting proofs of membership of $x$ in $\mathcal{L}$. Our result has applications to unclonable signatures of knowledge, which we define and construct in this work; these non-interactively prevent replay attacks.



## **9. Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders**

cs.LG

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08571v1) [paper-pdf](http://arxiv.org/pdf/2310.08571v1)

**Authors**: Jan Dubiński, Stanisław Pawlak, Franziska Boenisch, Tomasz Trzciński, Adam Dziedzic

**Abstract**: Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task.vB4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.



## **10. Jailbreaking Black Box Large Language Models in Twenty Queries**

cs.LG

21 pages, 10 figures

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08419v1) [paper-pdf](http://arxiv.org/pdf/2310.08419v1)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and PaLM-2.



## **11. Vault: Decentralized Storage Made Durable**

cs.DC

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08403v1) [paper-pdf](http://arxiv.org/pdf/2310.08403v1)

**Authors**: Guangda Sun, Michael Hu Yiqing, Arun Fu, Akasha Zhu, Jialin Li

**Abstract**: The lack of centralized control, combined with highly dynamic adversarial behaviors, makes data durability a challenge in decentralized storage systems. In this work, we introduce a new storage system, Vault, that offers strong data durability guarantees in a fully decentralized, permission-less setting. Vault leverages the rateless property of erasure code to encode each data object into an infinite stream of encoding fragments. To ensure durability in the presence of dynamic Byzantine behaviors and targeted attacks, an infinite sequence of storage nodes are randomly selected to store encoding fragments. Encoding generation and candidate selection are fully decentralized: When necessary, Vault nodes use a gossip protocol and a publically verifiable selection proof to determine new fragments. Simulations and large-scale EC2 experiments demonstrate that Vault provides close-to-ideal mean-time-to-data-loss (MTTDL) with low storage redundancy, scales to more than 10,000 nodes, and attains performance comparable to IPFS



## **12. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

cs.SD

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.05354v2) [paper-pdf](http://arxiv.org/pdf/2310.05354v2)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.



## **13. Defending Our Privacy With Backdoors**

cs.LG

14 pages, 4 figures

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08320v1) [paper-pdf](http://arxiv.org/pdf/2310.08320v1)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspective on backdoor attacks, but also presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.



## **14. Concealed Electronic Countermeasures of Radar Signal with Adversarial Examples**

eess.SP

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08292v1) [paper-pdf](http://arxiv.org/pdf/2310.08292v1)

**Authors**: Ruinan Ma, Canjie Zhu, Mingfeng Lu, Yunjie Li, Yu-an Tan, Ruibin Zhang, Ran Tao

**Abstract**: Electronic countermeasures involving radar signals are an important aspect of modern warfare. Traditional electronic countermeasures techniques typically add large-scale interference signals to ensure interference effects, which can lead to attacks being too obvious. In recent years, AI-based attack methods have emerged that can effectively solve this problem, but the attack scenarios are currently limited to time domain radar signal classification. In this paper, we focus on the time-frequency images classification scenario of radar signals. We first propose an attack pipeline under the time-frequency images scenario and DITIMI-FGSM attack algorithm with high transferability. Then, we propose STFT-based time domain signal attack(STDS) algorithm to solve the problem of non-invertibility in time-frequency analysis, thus obtaining the time-domain representation of the interference signal. A large number of experiments show that our attack pipeline is feasible and the proposed attack method has a high success rate.



## **15. Improving Fast Minimum-Norm Attacks with Hyperparameter Optimization**

cs.LG

Accepted at ESANN23

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08177v1) [paper-pdf](http://arxiv.org/pdf/2310.08177v1)

**Authors**: Giuseppe Floris, Raffaele Mura, Luca Scionis, Giorgio Piras, Maura Pintor, Ambra Demontis, Battista Biggio

**Abstract**: Evaluating the adversarial robustness of machine learning models using gradient-based attacks is challenging. In this work, we show that hyperparameter optimization can improve fast minimum-norm attacks by automating the selection of the loss function, the optimizer and the step-size scheduler, along with the corresponding hyperparameters. Our extensive evaluation involving several robust models demonstrates the improved efficacy of fast minimum-norm attacks when hyper-up with hyperparameter optimization. We release our open-source code at https://github.com/pralab/HO-FMN.



## **16. Samples on Thin Ice: Re-Evaluating Adversarial Pruning of Neural Networks**

cs.LG

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08073v1) [paper-pdf](http://arxiv.org/pdf/2310.08073v1)

**Authors**: Giorgio Piras, Maura Pintor, Ambra Demontis, Battista Biggio

**Abstract**: Neural network pruning has shown to be an effective technique for reducing the network size, trading desirable properties like generalization and robustness to adversarial attacks for higher sparsity. Recent work has claimed that adversarial pruning methods can produce sparse networks while also preserving robustness to adversarial examples. In this work, we first re-evaluate three state-of-the-art adversarial pruning methods, showing that their robustness was indeed overestimated. We then compare pruned and dense versions of the same models, discovering that samples on thin ice, i.e., closer to the unpruned model's decision boundary, are typically misclassified after pruning. We conclude by discussing how this intuition may lead to designing more effective adversarial pruning methods in future work.



## **17. Block Coordinate Descent on Smooth Manifolds: Convergence Theory and Twenty-One Examples**

math.OC

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2305.14744v3) [paper-pdf](http://arxiv.org/pdf/2305.14744v3)

**Authors**: Liangzu Peng, René Vidal

**Abstract**: Block coordinate descent is an optimization paradigm that iteratively updates one block of variables at a time, making it quite amenable to big data applications due to its scalability and performance. Its convergence behavior has been extensively studied in the (block-wise) convex case, but it is much less explored in the non-convex case. In this paper we analyze the convergence of block coordinate methods on non-convex sets and derive convergence rates on smooth manifolds under natural or weaker assumptions than prior work. Our analysis applies to many non-convex problems, including ones that seek low-dimensional structures (e.g., maximal coding rate reduction, neural collapse, reverse engineering adversarial attacks, generalized PCA, alternating projection); ones that seek combinatorial structures (homomorphic sensing, regression without correspondences, real phase retrieval, robust point matching); ones that seek geometric structures from visual data (e.g., essential matrix estimation, absolute pose estimation); and ones that seek inliers sparsely hidden in a large number of outliers (e.g., outlier-robust estimation via iteratively-reweighted least-squares). While our convergence theory applies to all these problems, yielding novel corollaries, it also applies to other, perhaps more familiar, problems (e.g., optimal transport, matrix factorization, Burer-Monteiro factorization), recovering previously known results.



## **18. Why Train More? Effective and Efficient Membership Inference via Memorization**

cs.LG

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08015v1) [paper-pdf](http://arxiv.org/pdf/2310.08015v1)

**Authors**: Jihye Choi, Shruti Tople, Varun Chandrasekaran, Somesh Jha

**Abstract**: Membership Inference Attacks (MIAs) aim to identify specific data samples within the private training dataset of machine learning models, leading to serious privacy violations and other sophisticated threats. Many practical black-box MIAs require query access to the data distribution (the same distribution where the private data is drawn) to train shadow models. By doing so, the adversary obtains models trained "with" or "without" samples drawn from the distribution, and analyzes the characteristics of the samples under consideration. The adversary is often required to train more than hundreds of shadow models to extract the signals needed for MIAs; this becomes the computational overhead of MIAs. In this paper, we propose that by strategically choosing the samples, MI adversaries can maximize their attack success while minimizing the number of shadow models. First, our motivational experiments suggest memorization as the key property explaining disparate sample vulnerability to MIAs. We formalize this through a theoretical bound that connects MI advantage with memorization. Second, we show sample complexity bounds that connect the number of shadow models needed for MIAs with memorization. Lastly, we confirm our theoretical arguments with comprehensive experiments; by utilizing samples with high memorization scores, the adversary can (a) significantly improve its efficacy regardless of the MIA used, and (b) reduce the number of shadow models by nearly two orders of magnitude compared to state-of-the-art approaches.



## **19. GRASP: Accelerating Shortest Path Attacks via Graph Attention**

cs.LG

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.07980v1) [paper-pdf](http://arxiv.org/pdf/2310.07980v1)

**Authors**: Zohair Shafi. Benjamin A. Miller, Ayan Chatterjee, Tina Eliassi-Rad, Rajmonda S. Caceres

**Abstract**: Recent advances in machine learning (ML) have shown promise in aiding and accelerating classical combinatorial optimization algorithms. ML-based speed ups that aim to learn in an end to end manner (i.e., directly output the solution) tend to trade off run time with solution quality. Therefore, solutions that are able to accelerate existing solvers while maintaining their performance guarantees, are of great interest. We consider an APX-hard problem, where an adversary aims to attack shortest paths in a graph by removing the minimum number of edges. We propose the GRASP algorithm: Graph Attention Accelerated Shortest Path Attack, an ML aided optimization algorithm that achieves run times up to 10x faster, while maintaining the quality of solution generated. GRASP uses a graph attention network to identify a smaller subgraph containing the combinatorial solution, thus effectively reducing the input problem size. Additionally, we demonstrate how careful representation of the input graph, including node features that correlate well with the optimization task, can highlight important structure in the optimization solution.



## **20. Multi-SpacePhish: Extending the Evasion-space of Adversarial Attacks against Phishing Website Detectors using Machine Learning**

cs.CR

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2210.13660v3) [paper-pdf](http://arxiv.org/pdf/2210.13660v3)

**Authors**: Ying Yuan, Giovanni Apruzzese, Mauro Conti

**Abstract**: Existing literature on adversarial Machine Learning (ML) focuses either on showing attacks that break every ML model, or defenses that withstand most attacks. Unfortunately, little consideration is given to the actual feasibility of the attack or the defense. Moreover, adversarial samples are often crafted in the "feature-space", making the corresponding evaluations of questionable value. Simply put, the current situation does not allow to estimate the actual threat posed by adversarial attacks, leading to a lack of secure ML systems.   We aim to clarify such confusion in this paper. By considering the application of ML for Phishing Website Detection (PWD), we formalize the "evasion-space" in which an adversarial perturbation can be introduced to fool a ML-PWD -- demonstrating that even perturbations in the "feature-space" are useful. Then, we propose a realistic threat model describing evasion attacks against ML-PWD that are cheap to stage, and hence intrinsically more attractive for real phishers. After that, we perform the first statistically validated assessment of state-of-the-art ML-PWD against 12 evasion attacks. Our evaluation shows (i) the true efficacy of evasion attempts that are more likely to occur; and (ii) the impact of perturbations crafted in different evasion-spaces. Our realistic evasion attempts induce a statistically significant degradation (3-10% at p<0.05), and their cheap cost makes them a subtle threat. Notably, however, some ML-PWD are immune to our most realistic attacks (p=0.22).   Finally, as an additional contribution of this journal publication, we are the first to consider the intriguing case wherein an attacker introduces perturbations in multiple evasion-spaces at the same time. These new results show that simultaneously applying perturbations in the problem- and feature-space can cause a drop in the detection rate from 0.95 to 0.



## **21. Deep Reinforcement Learning for Autonomous Cyber Operations: A Survey**

cs.LG

60 pages, 14 figures, 3 tables

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07745v1) [paper-pdf](http://arxiv.org/pdf/2310.07745v1)

**Authors**: Gregory Palmer, Chris Parry, Daniel J. B. Harrold, Chris Willis

**Abstract**: The rapid increase in the number of cyber-attacks in recent years raises the need for principled methods for defending networks against malicious actors. Deep reinforcement learning (DRL) has emerged as a promising approach for mitigating these attacks. However, while DRL has shown much potential for cyber-defence, numerous challenges must be overcome before DRL can be applied to autonomous cyber-operations (ACO) at scale. Principled methods are required for environments that confront learners with very high-dimensional state spaces, large multi-discrete action spaces, and adversarial learning. Recent works have reported success in solving these problems individually. There have also been impressive engineering efforts towards solving all three for real-time strategy games. However, applying DRL to the full ACO problem remains an open challenge. Here, we survey the relevant DRL literature and conceptualize an idealised ACO-DRL agent. We provide: i.) A summary of the domain properties that define the ACO problem; ii.) A comprehensive evaluation of the extent to which domains used for benchmarking DRL approaches are comparable to ACO; iii.) An overview of state-of-the-art approaches for scaling DRL to domains that confront learners with the curse of dimensionality, and; iv.) A survey and critique of current methods for limiting the exploitability of agents within adversarial settings from the perspective of ACO. We conclude with open research questions that we hope will motivate future directions for researchers and practitioners working on ACO.



## **22. Boosting Black-box Attack to Deep Neural Networks with Conditional Diffusion Models**

cs.CV

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07492v1) [paper-pdf](http://arxiv.org/pdf/2310.07492v1)

**Authors**: Renyang Liu, Wei Zhou, Tianwei Zhang, Kangjie Chen, Jun Zhao, Kwok-Yan Lam

**Abstract**: Existing black-box attacks have demonstrated promising potential in creating adversarial examples (AE) to deceive deep learning models. Most of these attacks need to handle a vast optimization space and require a large number of queries, hence exhibiting limited practical impacts in real-world scenarios. In this paper, we propose a novel black-box attack strategy, Conditional Diffusion Model Attack (CDMA), to improve the query efficiency of generating AEs under query-limited situations. The key insight of CDMA is to formulate the task of AE synthesis as a distribution transformation problem, i.e., benign examples and their corresponding AEs can be regarded as coming from two distinctive distributions and can transform from each other with a particular converter. Unlike the conventional \textit{query-and-optimization} approach, we generate eligible AEs with direct conditional transform using the aforementioned data converter, which can significantly reduce the number of queries needed. CDMA adopts the conditional Denoising Diffusion Probabilistic Model as the converter, which can learn the transformation from clean samples to AEs, and ensure the smooth development of perturbed noise resistant to various defense strategies. We demonstrate the effectiveness and efficiency of CDMA by comparing it with nine state-of-the-art black-box attacks across three benchmark datasets. On average, CDMA can reduce the query count to a handful of times; in most cases, the query count is only ONE. We also show that CDMA can obtain $>99\%$ attack success rate for untarget attacks over all datasets and targeted attack over CIFAR-10 with the noise budget of $\epsilon=16$.



## **23. Fundamental Limitations of Alignment in Large Language Models**

cs.CL

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2304.11082v4) [paper-pdf](http://arxiv.org/pdf/2304.11082v4)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.



## **24. My Brother Helps Me: Node Injection Based Adversarial Attack on Social Bot Detection**

cs.CR

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07159v1) [paper-pdf](http://arxiv.org/pdf/2310.07159v1)

**Authors**: Lanjun Wang, Xinran Qiao, Yanwei Xie, Weizhi Nie, Yongdong Zhang, Anan Liu

**Abstract**: Social platforms such as Twitter are under siege from a multitude of fraudulent users. In response, social bot detection tasks have been developed to identify such fake users. Due to the structure of social networks, the majority of methods are based on the graph neural network(GNN), which is susceptible to attacks. In this study, we propose a node injection-based adversarial attack method designed to deceive bot detection models. Notably, neither the target bot nor the newly injected bot can be detected when a new bot is added around the target bot. This attack operates in a black-box fashion, implying that any information related to the victim model remains unknown. To our knowledge, this is the first study exploring the resilience of bot detection through graph node injection. Furthermore, we develop an attribute recovery module to revert the injected node embedding from the graph embedding space back to the original feature space, enabling the adversary to manipulate node perturbation effectively. We conduct adversarial attacks on four commonly used GNN structures for bot detection on two widely used datasets: Cresci-2015 and TwiBot-22. The attack success rate is over 73\% and the rate of newly injected nodes being detected as bots is below 13\% on these two datasets.



## **25. No Privacy Left Outside: On the (In-)Security of TEE-Shielded DNN Partition for On-Device ML**

cs.CR

Accepted by S&P'24

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07152v1) [paper-pdf](http://arxiv.org/pdf/2310.07152v1)

**Authors**: Ziqi Zhang, Chen Gong, Yifeng Cai, Yuanyuan Yuan, Bingyan Liu, Ding Li, Yao Guo, Xiangqun Chen

**Abstract**: On-device ML introduces new security challenges: DNN models become white-box accessible to device users. Based on white-box information, adversaries can conduct effective model stealing (MS) and membership inference attack (MIA). Using Trusted Execution Environments (TEEs) to shield on-device DNN models aims to downgrade (easy) white-box attacks to (harder) black-box attacks. However, one major shortcoming is the sharply increased latency (up to 50X). To accelerate TEE-shield DNN computation with GPUs, researchers proposed several model partition techniques. These solutions, referred to as TEE-Shielded DNN Partition (TSDP), partition a DNN model into two parts, offloading the privacy-insensitive part to the GPU while shielding the privacy-sensitive part within the TEE. This paper benchmarks existing TSDP solutions using both MS and MIA across a variety of DNN models, datasets, and metrics. We show important findings that existing TSDP solutions are vulnerable to privacy-stealing attacks and are not as safe as commonly believed. We also unveil the inherent difficulty in deciding optimal DNN partition configurations (i.e., the highest security with minimal utility cost) for present TSDP solutions. The experiments show that such ``sweet spot'' configurations vary across datasets and models. Based on lessons harvested from the experiments, we present TEESlice, a novel TSDP method that defends against MS and MIA during DNN inference. TEESlice follows a partition-before-training strategy, which allows for accurate separation between privacy-related weights from public weights. TEESlice delivers the same security protection as shielding the entire DNN model inside TEE (the ``upper-bound'' security guarantees) with over 10X less overhead (in both experimental and real-world environments) than prior TSDP solutions and no accuracy loss.



## **26. Investigating the Adversarial Robustness of Density Estimation Using the Probability Flow ODE**

cs.LG

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.07084v1) [paper-pdf](http://arxiv.org/pdf/2310.07084v1)

**Authors**: Marius Arvinte, Cory Cornelius, Jason Martin, Nageen Himayat

**Abstract**: Beyond their impressive sampling capabilities, score-based diffusion models offer a powerful analysis tool in the form of unbiased density estimation of a query sample under the training data distribution. In this work, we investigate the robustness of density estimation using the probability flow (PF) neural ordinary differential equation (ODE) model against gradient-based likelihood maximization attacks and the relation to sample complexity, where the compressed size of a sample is used as a measure of its complexity. We introduce and evaluate six gradient-based log-likelihood maximization attacks, including a novel reverse integration attack. Our experimental evaluations on CIFAR-10 show that density estimation using the PF ODE is robust against high-complexity, high-likelihood attacks, and that in some cases adversarial samples are semantically meaningful, as expected from a robust estimator.



## **27. Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models**

cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2307.14539v2) [paper-pdf](http://arxiv.org/pdf/2307.14539v2)

**Authors**: Erfan Shayegani, Yue Dong, Nael Abu-Ghazaleh

**Abstract**: We introduce new jailbreak attacks on vision language models (VLMs), which use aligned LLMs and are resilient to text-only jailbreak attacks. Specifically, we develop cross-modality attacks on alignment where we pair adversarial images going through the vision encoder with textual prompts to break the alignment of the language model. Our attacks employ a novel compositional strategy that combines an image, adversarially targeted towards toxic embeddings, with generic prompts to accomplish the jailbreak. Thus, the LLM draws the context to answer the generic prompt from the adversarial image. The generation of benign-appearing adversarial images leverages a novel embedding-space-based methodology, operating with no access to the LLM model. Instead, the attacks require access only to the vision encoder and utilize one of our four embedding space targeting strategies. By not requiring access to the LLM, the attacks lower the entry barrier for attackers, particularly when vision encoders such as CLIP are embedded in closed-source LLMs. The attacks achieve a high success rate across different VLMs, highlighting the risk of cross-modality alignment vulnerabilities, and the need for new alignment approaches for multi-modal models.



## **28. TDPP: Two-Dimensional Permutation-Based Protection of Memristive Deep Neural Networks**

cs.CR

14 pages, 11 figures

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06989v1) [paper-pdf](http://arxiv.org/pdf/2310.06989v1)

**Authors**: Minhui Zou, Zhenhua Zhu, Tzofnat Greenberg-Toledo, Orian Leitersdorf, Jiang Li, Junlong Zhou, Yu Wang, Nan Du, Shahar Kvatinsky

**Abstract**: The execution of deep neural network (DNN) algorithms suffers from significant bottlenecks due to the separation of the processing and memory units in traditional computer systems. Emerging memristive computing systems introduce an in situ approach that overcomes this bottleneck. The non-volatility of memristive devices, however, may expose the DNN weights stored in memristive crossbars to potential theft attacks. Therefore, this paper proposes a two-dimensional permutation-based protection (TDPP) method that thwarts such attacks. We first introduce the underlying concept that motivates the TDPP method: permuting both the rows and columns of the DNN weight matrices. This contrasts with previous methods, which focused solely on permuting a single dimension of the weight matrices, either the rows or columns. While it's possible for an adversary to access the matrix values, the original arrangement of rows and columns in the matrices remains concealed. As a result, the extracted DNN model from the accessed matrix values would fail to operate correctly. We consider two different memristive computing systems (designed for layer-by-layer and layer-parallel processing, respectively) and demonstrate the design of the TDPP method that could be embedded into the two systems. Finally, we present a security analysis. Our experiments demonstrate that TDPP can achieve comparable effectiveness to prior approaches, with a high level of security when appropriately parameterized. In addition, TDPP is more scalable than previous methods and results in reduced area and power overheads. The area and power are reduced by, respectively, 1218$\times$ and 2815$\times$ for the layer-by-layer system and by 178$\times$ and 203$\times$ for the layer-parallel system compared to prior works.



## **29. Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation**

cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06987v1) [paper-pdf](http://arxiv.org/pdf/2310.06987v1)

**Authors**: Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, Danqi Chen

**Abstract**: The rapid progress in open-source large language models (LLMs) is significantly advancing AI development. Extensive efforts have been made before model release to align their behavior with human values, with the primary goal of ensuring their helpfulness and harmlessness. However, even carefully aligned models can be manipulated maliciously, leading to unintended behaviors, known as "jailbreaks". These jailbreaks are typically triggered by specific text inputs, often referred to as adversarial prompts. In this work, we propose the generation exploitation attack, an extremely simple approach that disrupts model alignment by only manipulating variations of decoding methods. By exploiting different generation strategies, including varying decoding hyper-parameters and sampling methods, we increase the misalignment rate from 0% to more than 95% across 11 language models including LLaMA2, Vicuna, Falcon, and MPT families, outperforming state-of-the-art attacks with $30\times$ lower computational cost. Finally, we propose an effective alignment method that explores diverse generation strategies, which can reasonably reduce the misalignment rate under our attack. Altogether, our study underscores a major failure in current safety evaluation and alignment procedures for open-source LLMs, strongly advocating for more comprehensive red teaming and better alignment before releasing such models. Our code is available at https://github.com/Princeton-SysML/Jailbreak_LLM.



## **30. Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks**

cs.CV

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06958v1) [paper-pdf](http://arxiv.org/pdf/2310.06958v1)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Sergey Lavrushkin, Ekaterina Shumitskaya, Maksim Velikanov, Dmitriy Vatolin

**Abstract**: Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. Try our benchmark using pip install robustness-benchmark.



## **31. Adversarial optimization leads to over-optimistic security-constrained dispatch, but sampling can help**

eess.SY

Accepted at NAPS 2023

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06956v1) [paper-pdf](http://arxiv.org/pdf/2310.06956v1)

**Authors**: Charles Dawson, Chuchu Fan

**Abstract**: To ensure safe, reliable operation of the electrical grid, we must be able to predict and mitigate likely failures. This need motivates the classic security-constrained AC optimal power flow (SCOPF) problem. SCOPF is commonly solved using adversarial optimization, where the dispatcher and an adversary take turns optimizing a robust dispatch and adversarial attack, respectively. We show that adversarial optimization is liable to severely overestimate the robustness of the optimized dispatch (when the adversary encounters a local minimum), leading the operator to falsely believe that their dispatch is secure.   To prevent this overconfidence, we develop a novel adversarial sampling approach that prioritizes diversity in the predicted attacks. We find that our method not only substantially improves the robustness of the optimized dispatch but also avoids overconfidence, accurately characterizing the likelihood of voltage collapse under a given threat model. We demonstrate a proof-of-concept on small-scale transmission systems with 14 and 57 nodes.



## **32. Graph-based methods coupled with specific distributional distances for adversarial attack detection**

cs.LG

published in Neural Networks

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2306.00042v2) [paper-pdf](http://arxiv.org/pdf/2306.00042v2)

**Authors**: Dwight Nwaigwe, Lucrezia Carboni, Martial Mermillod, Sophie Achard, Michel Dojat

**Abstract**: Artificial neural networks are prone to being fooled by carefully perturbed inputs which cause an egregious misclassification. These \textit{adversarial} attacks have been the focus of extensive research. Likewise, there has been an abundance of research in ways to detect and defend against them. We introduce a novel approach of detection and interpretation of adversarial attacks from a graph perspective. For an input image, we compute an associated sparse graph using the layer-wise relevance propagation algorithm \cite{bach15}. Specifically, we only keep edges of the neural network with the highest relevance values. Three quantities are then computed from the graph which are then compared against those computed from the training set. The result of the comparison is a classification of the image as benign or adversarial. To make the comparison, two classification methods are introduced: 1) an explicit formula based on Wasserstein distance applied to the degree of node and 2) a logistic regression. Both classification methods produce strong results which lead us to believe that a graph-based interpretation of adversarial attacks is valuable.



## **33. Privacy-oriented manipulation of speaker representations**

eess.AS

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06652v1) [paper-pdf](http://arxiv.org/pdf/2310.06652v1)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstract**: Speaker embeddings are ubiquitous, with applications ranging from speaker recognition and diarization to speech synthesis and voice anonymisation. The amount of information held by these embeddings lends them versatility, but also raises privacy concerns. Speaker embeddings have been shown to contain information on age, sex, health and more, which speakers may want to keep private, especially when this information is not required for the target task. In this work, we propose a method for removing and manipulating private attributes from speaker embeddings that leverages a Vector-Quantized Variational Autoencoder architecture, combined with an adversarial classifier and a novel mutual information loss. We validate our model on two attributes, sex and age, and perform experiments with ignorant and fully-informed attackers, and with in-domain and out-of-domain data.



## **34. A Geometrical Approach to Evaluate the Adversarial Robustness of Deep Neural Networks**

cs.CV

ACM Transactions on Multimedia Computing, Communications, and  Applications (ACM TOMM)

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06468v1) [paper-pdf](http://arxiv.org/pdf/2310.06468v1)

**Authors**: Yang Wang, Bo Dong, Ke Xu, Haiyin Piao, Yufei Ding, Baocai Yin, Xin Yang

**Abstract**: Deep Neural Networks (DNNs) are widely used for computer vision tasks. However, it has been shown that deep models are vulnerable to adversarial attacks, i.e., their performances drop when imperceptible perturbations are made to the original inputs, which may further degrade the following visual tasks or introduce new problems such as data and privacy security. Hence, metrics for evaluating the robustness of deep models against adversarial attacks are desired. However, previous metrics are mainly proposed for evaluating the adversarial robustness of shallow networks on the small-scale datasets. Although the Cross Lipschitz Extreme Value for nEtwork Robustness (CLEVER) metric has been proposed for large-scale datasets (e.g., the ImageNet dataset), it is computationally expensive and its performance relies on a tractable number of samples. In this paper, we propose the Adversarial Converging Time Score (ACTS), an attack-dependent metric that quantifies the adversarial robustness of a DNN on a specific input. Our key observation is that local neighborhoods on a DNN's output surface would have different shapes given different inputs. Hence, given different inputs, it requires different time for converging to an adversarial sample. Based on this geometry meaning, ACTS measures the converging time as an adversarial robustness metric. We validate the effectiveness and generalization of the proposed ACTS metric against different adversarial attacks on the large-scale ImageNet dataset using state-of-the-art deep networks. Extensive experiments show that our ACTS metric is an efficient and effective adversarial metric over the previous CLEVER metric.



## **35. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.00322v2) [paper-pdf](http://arxiv.org/pdf/2310.00322v2)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.



## **36. Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach**

cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS), New Orleans, USA, Dec. 2023, spotlight

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06396v1) [paper-pdf](http://arxiv.org/pdf/2310.06396v1)

**Authors**: Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, including those that affect both node features and graph topology. This paper investigates GNNs derived from diverse neural flows, concentrating on their connection to various stability notions such as BIBO stability, Lyapunov stability, structural stability, and conservative stability. We argue that Lyapunov stability, despite its common use, does not necessarily ensure adversarial robustness. Inspired by physics principles, we advocate for the use of conservative Hamiltonian neural flows to construct GNNs that are robust to adversarial attacks. The adversarial robustness of different neural flow GNNs is empirically compared on several benchmark datasets under a variety of adversarial attacks. Extensive numerical experiments demonstrate that GNNs leveraging conservative Hamiltonian flows with Lyapunov stability substantially improve robustness against adversarial perturbations. The implementation code of experiments is available at https://github.com/zknus/NeurIPS-2023-HANG-Robustness.



## **37. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

cs.LG

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06387v1) [paper-pdf](http://arxiv.org/pdf/2310.06387v1)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating malicious content have emerged. In this paper, we explore the power of In-Context Learning (ICL) in manipulating the alignment ability of LLMs. We find that by providing just few in-context demonstrations without fine-tuning, LLMs can be manipulated to increase or decrease the probability of jailbreaking, i.e. answering malicious prompts. Based on these observations, we propose In-Context Attack (ICA) and In-Context Defense (ICD) methods for jailbreaking and guarding aligned language model purposes. ICA crafts malicious contexts to guide models in generating harmful outputs, while ICD enhances model robustness by demonstrations of rejecting to answer harmful prompts. Our experiments show the effectiveness of ICA and ICD in increasing or reducing the success rate of adversarial jailbreaking attacks. Overall, we shed light on the potential of ICL to influence LLM behavior and provide a new perspective for enhancing the safety and alignment of LLMs.



## **38. Double Public Key Signing Function Oracle Attack on EdDSA Software Implementations**

cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2308.15009v2) [paper-pdf](http://arxiv.org/pdf/2308.15009v2)

**Authors**: Sam Grierson, Konstantinos Chalkias, William J Buchanan, Leandros Maglaras

**Abstract**: EdDSA is a standardised elliptic curve digital signature scheme introduced to overcome some of the issues prevalent in the more established ECDSA standard. Due to the EdDSA standard specifying that the EdDSA signature be deterministic, if the signing function were to be used as a public key signing oracle for the attacker, the unforgeability notion of security of the scheme can be broken. This paper describes an attack against some of the most popular EdDSA implementations, which results in an adversary recovering the private key used during signing. With this recovered secret key, an adversary can sign arbitrary messages that would be seen as valid by the EdDSA verification function. A list of libraries with vulnerable APIs at the time of publication is provided. Furthermore, this paper provides two suggestions for securing EdDSA signing APIs against this vulnerability while it additionally discusses failed attempts to solve the issue.



## **39. Exploring adversarial attacks in federated learning for medical imaging**

cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06227v1) [paper-pdf](http://arxiv.org/pdf/2310.06227v1)

**Authors**: Erfan Darzi, Florian Dubost, N. M. Sijtsema, P. M. A van Ooijen

**Abstract**: Federated learning offers a privacy-preserving framework for medical image analysis but exposes the system to adversarial attacks. This paper aims to evaluate the vulnerabilities of federated learning networks in medical image analysis against such attacks. Employing domain-specific MRI tumor and pathology imaging datasets, we assess the effectiveness of known threat scenarios in a federated learning environment. Our tests reveal that domain-specific configurations can increase the attacker's success rate significantly. The findings emphasize the urgent need for effective defense mechanisms and suggest a critical re-evaluation of current security protocols in federated medical image analysis systems.



## **40. PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization**

cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.06182v1) [paper-pdf](http://arxiv.org/pdf/2310.06182v1)

**Authors**: Jiancong Xiao, Ruoyu Sun, Zhi-quan Luo

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. It is found empirically that adversarially robust generalization is crucial in establishing defense algorithms against adversarial attacks. Therefore, it is interesting to study the theoretical guarantee of robust generalization. This paper focuses on norm-based complexity, based on a PAC-Bayes approach (Neyshabur et al., 2017). The main challenge lies in extending the key ingredient, which is a weight perturbation bound in standard settings, to the robust settings. Existing attempts heavily rely on additional strong assumptions, leading to loose bounds. In this paper, we address this issue and provide a spectrally-normalized robust generalization bound for DNNs. Compared to existing bounds, our bound offers two significant advantages: Firstly, it does not depend on additional assumptions. Secondly, it is considerably tighter, aligning with the bounds of standard generalization. Therefore, our result provides a different perspective on understanding robust generalization: The mismatch terms between standard and robust generalization bounds shown in previous studies do not contribute to the poor robust generalization. Instead, these disparities solely due to mathematical issues. Finally, we extend the main result to adversarial robustness against general non-$\ell_p$ attacks and other neural network architectures.



## **41. Lessons Learned: Defending Against Property Inference Attacks**

cs.CR

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2205.08821v4) [paper-pdf](http://arxiv.org/pdf/2205.08821v4)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstract**: This work investigates and evaluates multiple defense strategies against property inference attacks (PIAs), a privacy attack against machine learning models. Given a trained machine learning model, PIAs aim to extract statistical properties of its underlying training data, e.g., reveal the ratio of men and women in a medical training data set. While for other privacy attacks like membership inference, a lot of research on defense mechanisms has been published, this is the first work focusing on defending against PIAs. With the primary goal of developing a generic mitigation strategy against white-box PIAs, we propose the novel approach property unlearning. Extensive experiments with property unlearning show that while it is very effective when defending target models against specific adversaries, property unlearning is not able to generalize, i.e., protect against a whole class of PIAs. To investigate the reasons behind this limitation, we present the results of experiments with the explainable AI tool LIME. They show how state-of-the-art property inference adversaries with the same objective focus on different parts of the target model. We further elaborate on this with a follow-up experiment, in which we use the visualization technique t-SNE to exhibit how severely statistical training data properties are manifested in machine learning models. Based on this, we develop the conjecture that post-training techniques like property unlearning might not suffice to provide the desirable generic protection against PIAs. As an alternative, we investigate the effects of simpler training data preprocessing methods like adding Gaussian noise to images of a training data set on the success rate of PIAs. We conclude with a discussion of the different defense approaches, summarize the lessons learned and provide directions for future work.



## **42. Universal adversarial perturbations for multiple classification tasks with quantum classifiers**

quant-ph

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2306.11974v2) [paper-pdf](http://arxiv.org/pdf/2306.11974v2)

**Authors**: Yun-Zhong Qiu

**Abstract**: Quantum adversarial machine learning is an emerging field that studies the vulnerability of quantum learning systems against adversarial perturbations and develops possible defense strategies. Quantum universal adversarial perturbations are small perturbations, which can make different input samples into adversarial examples that may deceive a given quantum classifier. This is a field that was rarely looked into but worthwhile investigating because universal perturbations might simplify malicious attacks to a large extent, causing unexpected devastation to quantum machine learning models. In this paper, we take a step forward and explore the quantum universal perturbations in the context of heterogeneous classification tasks. In particular, we find that quantum classifiers that achieve almost state-of-the-art accuracy on two different classification tasks can be both conclusively deceived by one carefully-crafted universal perturbation. This result is explicitly demonstrated with well-designed quantum continual learning models with elastic weight consolidation method to avoid catastrophic forgetting, as well as real-life heterogeneous datasets from hand-written digits and medical MRI images. Our results provide a simple and efficient way to generate universal perturbations on heterogeneous classification tasks and thus would provide valuable guidance for future quantum learning technologies.



## **43. RECESS Vaccine for Federated Learning: Proactive Defense Against Model Poisoning Attacks**

cs.CR

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05431v1) [paper-pdf](http://arxiv.org/pdf/2310.05431v1)

**Authors**: Haonan Yan, Wenjing Zhang, Qian Chen, Xiaoguang Li, Wenhai Sun, Hui Li, Xiaodong Lin

**Abstract**: Model poisoning attacks greatly jeopardize the application of federated learning (FL). The effectiveness of existing defenses is susceptible to the latest model poisoning attacks, leading to a decrease in prediction accuracy. Besides, these defenses are intractable to distinguish benign outliers from malicious gradients, which further compromises the model generalization. In this work, we propose a novel proactive defense named RECESS against model poisoning attacks. Different from the passive analysis in previous defenses, RECESS proactively queries each participating client with a delicately constructed aggregation gradient, accompanied by the detection of malicious clients according to their responses with higher accuracy. Furthermore, RECESS uses a new trust scoring mechanism to robustly aggregate gradients. Unlike previous methods that score each iteration, RECESS considers clients' performance correlation across multiple iterations to estimate the trust score, substantially increasing fault tolerance. Finally, we extensively evaluate RECESS on typical model architectures and four datasets under various settings. We also evaluated the defensive effectiveness against other types of poisoning attacks, the sensitivity of hyperparameters, and adaptive adversarial attacks. Experimental results show the superiority of RECESS in terms of reducing accuracy loss caused by the latest model poisoning attacks over five classic and two state-of-the-art defenses.



## **44. AdvSV: An Over-the-Air Adversarial Attack Dataset for Speaker Verification**

cs.SD

Submitted to ICASSP2024

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05369v1) [paper-pdf](http://arxiv.org/pdf/2310.05369v1)

**Authors**: Li Wang, Jiaqi Li, Yuhao Luo, Jiahao Zheng, Lei Wang, Hao Li, Ke Xu, Chengfang Fang, Jie Shi, Zhizheng Wu

**Abstract**: It is known that deep neural networks are vulnerable to adversarial attacks. Although Automatic Speaker Verification (ASV) built on top of deep neural networks exhibits robust performance in controlled scenarios, many studies confirm that ASV is vulnerable to adversarial attacks. The lack of a standard dataset is a bottleneck for further research, especially reproducible research. In this study, we developed an open-source adversarial attack dataset for speaker verification research. As an initial step, we focused on the over-the-air attack. An over-the-air adversarial attack involves a perturbation generation algorithm, a loudspeaker, a microphone, and an acoustic environment. The variations in the recording configurations make it very challenging to reproduce previous research. The AdvSV dataset is constructed using the Voxceleb1 Verification test set as its foundation. This dataset employs representative ASV models subjected to adversarial attacks and records adversarial samples to simulate over-the-air attack settings. The scope of the dataset can be easily extended to include more types of adversarial attacks. The dataset will be released to the public under the CC-BY license. In addition, we also provide a detection baseline for reproducible research.



## **45. GReAT: A Graph Regularized Adversarial Training Method**

cs.LG

25 pages including references. 7 figures and 4 tables

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05336v1) [paper-pdf](http://arxiv.org/pdf/2310.05336v1)

**Authors**: Samet Bayram, Kenneth Barner

**Abstract**: This paper proposes a regularization method called GReAT, Graph Regularized Adversarial Training, to improve deep learning models' classification performance. Adversarial examples are a well-known challenge in machine learning, where small, purposeful perturbations to input data can mislead models. Adversarial training, a powerful and one of the most effective defense strategies, involves training models with both regular and adversarial examples. However, it often neglects the underlying structure of the data. In response, we propose GReAT, a method that leverages data graph structure to enhance model robustness. GReAT deploys the graph structure of the data into the adversarial training process, resulting in more robust models that better generalize its testing performance and defend against adversarial attacks. Through extensive evaluation on benchmark datasets, we demonstrate GReAT's effectiveness compared to state-of-the-art classification methods, highlighting its potential in improving deep learning models' classification performance.



## **46. On the Query Complexity of Training Data Reconstruction in Private Learning**

cs.LG

Matching upper bounds, new corollaries for DP variants

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2303.16372v5) [paper-pdf](http://arxiv.org/pdf/2303.16372v5)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We analyze the number of queries that a whitebox adversary needs to make to a private learner in order to reconstruct its training data. For $(\epsilon, \delta)$ DP learners with training data drawn from any arbitrary compact metric space, we provide the \emph{first known lower bounds on the adversary's query complexity} as a function of the learner's privacy parameters. \emph{Our results are minimax optimal for every $\epsilon \geq 0, \delta \in [0, 1]$, covering both $\epsilon$-DP and $(0, \delta)$ DP as corollaries}. Beyond this, we obtain query complexity lower bounds for $(\alpha, \epsilon)$ R\'enyi DP learners that are valid for any $\alpha > 1, \epsilon \geq 0$. Finally, we analyze data reconstruction attacks on locally compact metric spaces via the framework of Metric DP, a generalization of DP that accounts for the underlying metric structure of the data. In this setting, we provide the first known analysis of data reconstruction in unbounded, high dimensional spaces and obtain query complexity lower bounds that are nearly tight modulo logarithmic factors.



## **47. Adversarial Attacks on Combinatorial Multi-Armed Bandits**

cs.LG

28 pages

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05308v1) [paper-pdf](http://arxiv.org/pdf/2310.05308v1)

**Authors**: Rishab Balasubramanian, Jiawei Li, Prasad Tadepalli, Huazheng Wang, Qingyun Wu, Haoyu Zhao

**Abstract**: We study reward poisoning attacks on Combinatorial Multi-armed Bandits (CMAB). We first provide a sufficient and necessary condition for the attackability of CMAB, which depends on the intrinsic properties of the corresponding CMAB instance such as the reward distributions of super arms and outcome distributions of base arms. Additionally, we devise an attack algorithm for attackable CMAB instances. Contrary to prior understanding of multi-armed bandits, our work reveals a surprising fact that the attackability of a specific CMAB instance also depends on whether the bandit instance is known or unknown to the adversary. This finding indicates that adversarial attacks on CMAB are difficult in practice and a general attack strategy for any CMAB instance does not exist since the environment is mostly unknown to the adversary. We validate our theoretical findings via extensive experiments on real-world CMAB applications including probabilistic maximum covering problem, online minimum spanning tree, cascading bandits for online ranking, and online shortest path.



## **48. Robust Lipschitz Bandits to Adversarial Corruptions**

cs.LG

Thirty-seventh Conference on Neural Information Processing Systems  (NeurIPS 2023)

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2305.18543v2) [paper-pdf](http://arxiv.org/pdf/2305.18543v2)

**Authors**: Yue Kang, Cho-Jui Hsieh, Thomas C. M. Lee

**Abstract**: Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.



## **49. Susceptibility of Continual Learning Against Adversarial Attacks**

cs.LG

18 pages, 13 figures

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2207.05225v5) [paper-pdf](http://arxiv.org/pdf/2207.05225v5)

**Authors**: Hikmat Khan, Pir Masoom Shah, Syed Farhan Alam Zaidi, Saif ul Islam, Qasim Zia

**Abstract**: Recent continual learning approaches have primarily focused on mitigating catastrophic forgetting. Nevertheless, two critical areas have remained relatively unexplored: 1) evaluating the robustness of proposed methods and 2) ensuring the security of learned tasks. This paper investigates the susceptibility of continually learned tasks, including current and previously acquired tasks, to adversarial attacks. Specifically, we have observed that any class belonging to any task can be easily targeted and misclassified as the desired target class of any other task. Such susceptibility or vulnerability of learned tasks to adversarial attacks raises profound concerns regarding data integrity and privacy. To assess the robustness of continual learning approaches, we consider continual learning approaches in all three scenarios, i.e., task-incremental learning, domain-incremental learning, and class-incremental learning. In this regard, we explore the robustness of three regularization-based methods, three replay-based approaches, and one hybrid technique that combines replay and exemplar approaches. We empirically demonstrated that in any setting of continual learning, any class, whether belonging to the current or previously learned tasks, is susceptible to misclassification. Our observations identify potential limitations of continual learning approaches against adversarial attacks and highlight that current continual learning algorithms could not be suitable for deployment in real-world settings.



## **50. Transferable Availability Poisoning Attacks**

cs.CR

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05141v1) [paper-pdf](http://arxiv.org/pdf/2310.05141v1)

**Authors**: Yiyong Liu, Michael Backes, Xiao Zhang

**Abstract**: We consider availability data poisoning attacks, where an adversary aims to degrade the overall test accuracy of a machine learning model by crafting small perturbations to its training data. Existing poisoning strategies can achieve the attack goal but assume the victim to employ the same learning method as what the adversary uses to mount the attack. In this paper, we argue that this assumption is strong, since the victim may choose any learning algorithm to train the model as long as it can achieve some targeted performance on clean data. Empirically, we observe a large decrease in the effectiveness of prior poisoning attacks if the victim uses a different learning paradigm to train the model and show marked differences in frequency-level characteristics between perturbations generated with respect to different learners and attack methods. To enhance the attack transferability, we propose Transferable Poisoning, which generates high-frequency poisoning perturbations by alternately leveraging the gradient information with two specific algorithms selected from supervised and unsupervised contrastive learning paradigms. Through extensive experiments on benchmark image datasets, we show that our transferable poisoning attack can produce poisoned samples with significantly improved transferability, not only applicable to the two learners used to devise the attack but also for learning algorithms and even paradigms beyond.



