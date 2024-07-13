# Latest Adversarial Attack Papers
**update at 2024-07-13 10:45:01**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. How to beat a Bayesian adversary**

cs.LG

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08678v1) [paper-pdf](http://arxiv.org/pdf/2407.08678v1)

**Authors**: Zihan Ding, Kexin Jin, Jonas Latz, Chenguang Liu

**Abstract**: Deep neural networks and other modern machine learning models are often susceptible to adversarial attacks. Indeed, an adversary may often be able to change a model's prediction through a small, directed perturbation of the model's input - an issue in safety-critical applications. Adversarially robust machine learning is usually based on a minmax optimisation problem that minimises the machine learning loss under maximisation-based adversarial attacks.   In this work, we study adversaries that determine their attack using a Bayesian statistical approach rather than maximisation. The resulting Bayesian adversarial robustness problem is a relaxation of the usual minmax problem. To solve this problem, we propose Abram - a continuous-time particle system that shall approximate the gradient flow corresponding to the underlying learning problem. We show that Abram approximates a McKean-Vlasov process and justify the use of Abram by giving assumptions under which the McKean-Vlasov process finds the minimiser of the Bayesian adversarial robustness problem. We discuss two ways to discretise Abram and show its suitability in benchmark adversarial deep learning experiments.



## **2. Large-Scale Dataset Pruning in Adversarial Training through Data Importance Extrapolation**

cs.LG

8 pages, 5 figures, 3 tables, to be published in ICML: DMLR workshop

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2406.13283v2) [paper-pdf](http://arxiv.org/pdf/2406.13283v2)

**Authors**: Björn Nieth, Thomas Altstidl, Leo Schwinn, Björn Eskofier

**Abstract**: Their vulnerability to small, imperceptible attacks limits the adoption of deep learning models to real-world systems. Adversarial training has proven to be one of the most promising strategies against these attacks, at the expense of a substantial increase in training time. With the ongoing trend of integrating large-scale synthetic data this is only expected to increase even further. Thus, the need for data-centric approaches that reduce the number of training samples while maintaining accuracy and robustness arises. While data pruning and active learning are prominent research topics in deep learning, they are as of now largely unexplored in the adversarial training literature. We address this gap and propose a new data pruning strategy based on extrapolating data importance scores from a small set of data to a larger set. In an empirical evaluation, we demonstrate that extrapolation-based pruning can efficiently reduce dataset size while maintaining robustness.



## **3. DART: A Solution for Decentralized Federated Learning Model Robustness Analysis**

cs.DC

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08652v1) [paper-pdf](http://arxiv.org/pdf/2407.08652v1)

**Authors**: Chao Feng, Alberto Huertas Celdrán, Jan von der Assen, Enrique Tomás Martínez Beltrán, Gérôme Bovet, Burkhard Stiller

**Abstract**: Federated Learning (FL) has emerged as a promising approach to address privacy concerns inherent in Machine Learning (ML) practices. However, conventional FL methods, particularly those following the Centralized FL (CFL) paradigm, utilize a central server for global aggregation, which exhibits limitations such as bottleneck and single point of failure. To address these issues, the Decentralized FL (DFL) paradigm has been proposed, which removes the client-server boundary and enables all participants to engage in model training and aggregation tasks. Nevertheless, as CFL, DFL remains vulnerable to adversarial attacks, notably poisoning attacks that undermine model performance. While existing research on model robustness has predominantly focused on CFL, there is a noteworthy gap in understanding the model robustness of the DFL paradigm. In this paper, a thorough review of poisoning attacks targeting the model robustness in DFL systems, as well as their corresponding countermeasures, are presented. Additionally, a solution called DART is proposed to evaluate the robustness of DFL models, which is implemented and integrated into a DFL platform. Through extensive experiments, this paper compares the behavior of CFL and DFL under diverse poisoning attacks, pinpointing key factors affecting attack spread and effectiveness within the DFL. It also evaluates the performance of different defense mechanisms and investigates whether defense mechanisms designed for CFL are compatible with DFL. The empirical results provide insights into research challenges and suggest ways to improve the robustness of DFL models for future research.



## **4. RAIFLE: Reconstruction Attacks on Interaction-based Federated Learning with Adversarial Data Manipulation**

cs.CR

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2310.19163v2) [paper-pdf](http://arxiv.org/pdf/2310.19163v2)

**Authors**: Dzung Pham, Shreyas Kulkarni, Amir Houmansadr

**Abstract**: Federated learning has emerged as a promising privacy-preserving solution for machine learning domains that rely on user interactions, particularly recommender systems and online learning to rank. While there has been substantial research on the privacy of traditional federated learning, little attention has been paid to the privacy properties of these interaction-based settings. In this work, we show that users face an elevated risk of having their private interactions reconstructed by the central server when the server can control the training features of the items that users interact with. We introduce RAIFLE, a novel optimization-based attack framework where the server actively manipulates the features of the items presented to users to increase the success rate of reconstruction. Our experiments with federated recommendation and online learning-to-rank scenarios demonstrate that RAIFLE is significantly more powerful than existing reconstruction attacks like gradient inversion, achieving high performance consistently in most settings. We discuss the pros and cons of several possible countermeasures to defend against RAIFLE in the context of interaction-based federated learning. Our code is open-sourced at https://github.com/dzungvpham/raifle.



## **5. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2402.08656v5) [paper-pdf](http://arxiv.org/pdf/2402.08656v5)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.



## **6. Boosting Adversarial Transferability for Skeleton-based Action Recognition via Exploring the Model Posterior Space**

cs.CV

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08572v1) [paper-pdf](http://arxiv.org/pdf/2407.08572v1)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Xun Yang, Meng Wang, He Wang

**Abstract**: Skeletal motion plays a pivotal role in human activity recognition (HAR). Recently, attack methods have been proposed to identify the universal vulnerability of skeleton-based HAR(S-HAR). However, the research of adversarial transferability on S-HAR is largely missing. More importantly, existing attacks all struggle in transfer across unknown S-HAR models. We observed that the key reason is that the loss landscape of the action recognizers is rugged and sharp. Given the established correlation in prior studies~\cite{qin2022boosting,wu2020towards} between loss landscape and adversarial transferability, we assume and empirically validate that smoothing the loss landscape could potentially improve adversarial transferability on S-HAR. This is achieved by proposing a new post-train Dual Bayesian strategy, which can effectively explore the model posterior space for a collection of surrogates without the need for re-training. Furthermore, to craft adversarial examples along the motion manifold, we incorporate the attack gradient with information of the motion dynamics in a Bayesian manner. Evaluated on benchmark datasets, e.g. HDM05 and NTU 60, the average transfer success rate can reach as high as 35.9\% and 45.5\% respectively. In comparison, current state-of-the-art skeletal attacks achieve only 3.6\% and 9.8\%. The high adversarial transferability remains consistent across various surrogate, victim, and even defense models. Through a comprehensive analysis of the results, we provide insights on what surrogates are more likely to exhibit transferability, to shed light on future research.



## **7. BriDe Arbitrager: Enhancing Arbitrage in Ethereum 2.0 via Bribery-enabled Delayed Block Production**

cs.NI

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08537v1) [paper-pdf](http://arxiv.org/pdf/2407.08537v1)

**Authors**: Hulin Yang, Mingzhe Li, Jin Zhang, Alia Asheralieva, Qingsong Wei, Siow Mong Rick Goh

**Abstract**: The advent of Ethereum 2.0 has introduced significant changes, particularly the shift to Proof-of-Stake consensus. This change presents new opportunities and challenges for arbitrage. Amidst these changes, we introduce BriDe Arbitrager, a novel tool designed for Ethereum 2.0 that leverages Bribery-driven attacks to Delay block production and increase arbitrage gains. The main idea is to allow malicious proposers to delay block production by bribing validators/proposers, thereby gaining more time to identify arbitrage opportunities. Through analysing the bribery process, we design an adaptive bribery strategy. Additionally, we propose a Delayed Transaction Ordering Algorithm to leverage the delayed time to amplify arbitrage profits for malicious proposers. To ensure fairness and automate the bribery process, we design and implement a bribery smart contract and a bribery client. As a result, BriDe Arbitrager enables adversaries controlling a limited (< 1/4) fraction of the voting powers to delay block production via bribery and arbitrage more profit. Extensive experimental results based on Ethereum historical transactions demonstrate that BriDe Arbitrager yields an average of 8.66 ETH (16,442.23 USD) daily profits. Furthermore, our approach does not trigger any slashing mechanisms and remains effective even under Proposer Builder Separation and other potential mechanisms will be adopted by Ethereum.



## **8. Rethinking the Threat and Accessibility of Adversarial Attacks against Face Recognition Systems**

cs.CV

19 pages, 12 figures

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08514v1) [paper-pdf](http://arxiv.org/pdf/2407.08514v1)

**Authors**: Yuxin Cao, Yumeng Zhu, Derui Wang, Sheng Wen, Minhui Xue, Jin Lu, Hao Ge

**Abstract**: Face recognition pipelines have been widely deployed in various mission-critical systems in trust, equitable and responsible AI applications. However, the emergence of adversarial attacks has threatened the security of the entire recognition pipeline. Despite the sheer number of attack methods proposed for crafting adversarial examples in both digital and physical forms, it is never an easy task to assess the real threat level of different attacks and obtain useful insight into the key risks confronted by face recognition systems. Traditional attacks view imperceptibility as the most important measurement to keep perturbations stealthy, while we suspect that industry professionals may possess a different opinion. In this paper, we delve into measuring the threat brought about by adversarial attacks from the perspectives of the industry and the applications of face recognition. In contrast to widely studied sophisticated attacks in the field, we propose an effective yet easy-to-launch physical adversarial attack, named AdvColor, against black-box face recognition pipelines in the physical world. AdvColor fools models in the recognition pipeline via directly supplying printed photos of human faces to the system under adversarial illuminations. Experimental results show that physical AdvColor examples can achieve a fooling rate of more than 96% against the anti-spoofing model and an overall attack success rate of 88% against the face recognition pipeline. We also conduct a survey on the threats of prevailing adversarial attacks, including AdvColor, to understand the gap between the machine-measured and human-assessed threat levels of different forms of adversarial attacks. The survey results surprisingly indicate that, compared to deliberately launched imperceptible attacks, perceptible but accessible attacks pose more lethal threats to real-world commercial systems of face recognition.



## **9. Resilience of Entropy Model in Distributed Neural Networks**

cs.LG

accepted at ECCV 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2403.00942v2) [paper-pdf](http://arxiv.org/pdf/2403.00942v2)

**Authors**: Milin Zhang, Mohammad Abdi, Shahriar Rifat, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have emerged as a key technique to reduce communication overhead without sacrificing performance in edge computing systems. Recently, entropy coding has been introduced to further reduce the communication overhead. The key idea is to train the distributed DNN jointly with an entropy model, which is used as side information during inference time to adaptively encode latent representations into bit streams with variable length. To the best of our knowledge, the resilience of entropy models is yet to be investigated. As such, in this paper we formulate and investigate the resilience of entropy models to intentional interference (e.g., adversarial attacks) and unintentional interference (e.g., weather changes and motion blur). Through an extensive experimental campaign with 3 different DNN architectures, 2 entropy models and 4 rate-distortion trade-off factors, we demonstrate that the entropy attacks can increase the communication overhead by up to 95%. By separating compression features in frequency and spatial domain, we propose a new defense mechanism that can reduce the transmission overhead of the attacked input by about 9% compared to unperturbed data, with only about 2% accuracy loss. Importantly, the proposed defense mechanism is a standalone approach which can be applied in conjunction with approaches such as adversarial training to further improve robustness. Code will be shared for reproducibility.



## **10. Shedding More Light on Robust Classifiers under the lens of Energy-based Models**

cs.CV

Accepted at European Conference on Computer Vision (ECCV) 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.06315v2) [paper-pdf](http://arxiv.org/pdf/2407.06315v2)

**Authors**: Mujtaba Hussain Mirza, Maria Rosaria Briglia, Senad Beadini, Iacopo Masi

**Abstract**: By reinterpreting a robust discriminative classifier as Energy-based Model (EBM), we offer a new take on the dynamics of adversarial training (AT). Our analysis of the energy landscape during AT reveals that untargeted attacks generate adversarial images much more in-distribution (lower energy) than the original data from the point of view of the model. Conversely, we observe the opposite for targeted attacks. On the ground of our thorough analysis, we present new theoretical and practical results that show how interpreting AT energy dynamics unlocks a better understanding: (1) AT dynamic is governed by three phases and robust overfitting occurs in the third phase with a drastic divergence between natural and adversarial energies (2) by rewriting the loss of TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization (TRADES) in terms of energies, we show that TRADES implicitly alleviates overfitting by means of aligning the natural energy with the adversarial one (3) we empirically show that all recent state-of-the-art robust classifiers are smoothing the energy landscape and we reconcile a variety of studies about understanding AT and weighting the loss function under the umbrella of EBMs. Motivated by rigorous evidence, we propose Weighted Energy Adversarial Training (WEAT), a novel sample weighting scheme that yields robust accuracy matching the state-of-the-art on multiple benchmarks such as CIFAR-10 and SVHN and going beyond in CIFAR-100 and Tiny-ImageNet. We further show that robust classifiers vary in the intensity and quality of their generative capabilities, and offer a simple method to push this capability, reaching a remarkable Inception Score (IS) and FID using a robust classifier without training for generative modeling. The code to reproduce our results is available at http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/ .



## **11. A Human-in-the-Middle Attack against Object Detection Systems**

cs.RO

Accepted by IEEE Transactions on Artificial Intelligence, 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2208.07174v4) [paper-pdf](http://arxiv.org/pdf/2208.07174v4)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Object detection systems using deep learning models have become increasingly popular in robotics thanks to the rising power of CPUs and GPUs in embedded systems. However, these models are susceptible to adversarial attacks. While some attacks are limited by strict assumptions on access to the detection system, we propose a novel hardware attack inspired by Man-in-the-Middle attacks in cryptography. This attack generates a Universal Adversarial Perturbations (UAP) and injects the perturbation between the USB camera and the detection system via a hardware attack. Besides, prior research is misled by an evaluation metric that measures the model accuracy rather than the attack performance. In combination with our proposed evaluation metrics, we significantly increased the strength of adversarial perturbations. These findings raise serious concerns for applications of deep learning models in safety-critical systems, such as autonomous driving.



## **12. Venomancer: Towards Imperceptible and Target-on-Demand Backdoor Attacks in Federated Learning**

cs.CV

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.03144v2) [paper-pdf](http://arxiv.org/pdf/2407.03144v2)

**Authors**: Son Nguyen, Thinh Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that maintains data privacy by training on decentralized data sources. Similar to centralized machine learning, FL is also susceptible to backdoor attacks, where an attacker can compromise some clients by injecting a backdoor trigger into local models of those clients, leading to the global model's behavior being manipulated as desired by the attacker. Most backdoor attacks in FL assume a predefined target class and require control over a large number of clients or knowledge of benign clients' information. Furthermore, they are not imperceptible and are easily detected by human inspection due to clear artifacts left on the poison data. To overcome these challenges, we propose Venomancer, an effective backdoor attack that is imperceptible and allows target-on-demand. Specifically, imperceptibility is achieved by using a visual loss function to make the poison data visually indistinguishable from the original data. Target-on-demand property allows the attacker to choose arbitrary target classes via conditional adversarial training. Additionally, experiments showed that the method is robust against state-of-the-art defenses such as Norm Clipping, Weak DP, Krum, Multi-Krum, RLR, FedRAD, Deepsight, and RFLBAT. The source code is available at https://github.com/nguyenhongson1902/Venomancer.



## **13. A Comprehensive Survey on the Security of Smart Grid: Challenges, Mitigations, and Future Research Opportunities**

cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07966v1) [paper-pdf](http://arxiv.org/pdf/2407.07966v1)

**Authors**: Arastoo Zibaeirad, Farnoosh Koleini, Shengping Bi, Tao Hou, Tao Wang

**Abstract**: In this study, we conduct a comprehensive review of smart grid security, exploring system architectures, attack methodologies, defense strategies, and future research opportunities. We provide an in-depth analysis of various attack vectors, focusing on new attack surfaces introduced by advanced components in smart grids. The review particularly includes an extensive analysis of coordinated attacks that incorporate multiple attack strategies and exploit vulnerabilities across various smart grid components to increase their adverse impact, demonstrating the complexity and potential severity of these threats. Following this, we examine innovative detection and mitigation strategies, including game theory, graph theory, blockchain, and machine learning, discussing their advancements in counteracting evolving threats and associated research challenges. In particular, our review covers a thorough examination of widely used machine learning-based mitigation strategies, analyzing their applications and research challenges spanning across supervised, unsupervised, semi-supervised, ensemble, and reinforcement learning. Further, we outline future research directions and explore new techniques and concerns. We first discuss the research opportunities for existing and emerging strategies, and then explore the potential role of new techniques, such as large language models (LLMs), and the emerging threat of adversarial machine learning in the future of smart grid security.



## **14. Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies**

cs.LG

ICML 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2404.09349v2) [paper-pdf](http://arxiv.org/pdf/2404.09349v2)

**Authors**: Brian R. Bartoldson, James Diffenderfer, Konstantinos Parasyris, Bhavya Kailkhura

**Abstract**: This paper revisits the simple, long-studied, yet still unsolved problem of making image classifiers robust to imperceptible perturbations. Taking CIFAR10 as an example, SOTA clean accuracy is about $100$%, but SOTA robustness to $\ell_{\infty}$-norm bounded perturbations barely exceeds $70$%. To understand this gap, we analyze how model size, dataset size, and synthetic data quality affect robustness by developing the first scaling laws for adversarial training. Our scaling laws reveal inefficiencies in prior art and provide actionable feedback to advance the field. For instance, we discovered that SOTA methods diverge notably from compute-optimal setups, using excess compute for their level of robustness. Leveraging a compute-efficient setup, we surpass the prior SOTA with $20$% ($70$%) fewer training (inference) FLOPs. We trained various compute-efficient models, with our best achieving $74$% AutoAttack accuracy ($+3$% gain). However, our scaling laws also predict robustness slowly grows then plateaus at $90$%: dwarfing our new SOTA by scaling is impractical, and perfect robustness is impossible. To better understand this predicted limit, we carry out a small-scale human evaluation on the AutoAttack data that fools our top-performing model. Concerningly, we estimate that human performance also plateaus near $90$%, which we show to be attributable to $\ell_{\infty}$-constrained attacks' generation of invalid images not consistent with their original labels. Having characterized limiting roadblocks, we outline promising paths for future research.



## **15. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

cs.CV

ECCV2024. Code is available at  https://github.com/SensenGao/VLPTransferAttack

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2403.12445v2) [paper-pdf](http://arxiv.org/pdf/2403.12445v2)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs).Strengthening attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can advance reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability.In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs.To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods.Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks.



## **16. Targeted Augmented Data for Audio Deepfake Detection**

cs.SD

Accepted in EUSIPCO 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07598v1) [paper-pdf](http://arxiv.org/pdf/2407.07598v1)

**Authors**: Marcella Astrid, Enjie Ghorbel, Djamila Aouada

**Abstract**: The availability of highly convincing audio deepfake generators highlights the need for designing robust audio deepfake detectors. Existing works often rely solely on real and fake data available in the training set, which may lead to overfitting, thereby reducing the robustness to unseen manipulations. To enhance the generalization capabilities of audio deepfake detectors, we propose a novel augmentation method for generating audio pseudo-fakes targeting the decision boundary of the model. Inspired by adversarial attacks, we perturb original real data to synthesize pseudo-fakes with ambiguous prediction probabilities. Comprehensive experiments on two well-known architectures demonstrate that the proposed augmentation contributes to improving the generalization capabilities of these architectures.



## **17. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2305.17000v5) [paper-pdf](http://arxiv.org/pdf/2305.17000v5)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **18. Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models**

cs.CL

COLM 2024, 29 pages, 6 figures

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2405.15984v2) [paper-pdf](http://arxiv.org/pdf/2405.15984v2)

**Authors**: Simon Chi Lok Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl



## **19. Invisible Optical Adversarial Stripes on Traffic Sign against Autonomous Vehicles**

cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07510v1) [paper-pdf](http://arxiv.org/pdf/2407.07510v1)

**Authors**: Dongfang Guo, Yuting Wu, Yimin Dai, Pengfei Zhou, Xin Lou, Rui Tan

**Abstract**: Camera-based computer vision is essential to autonomous vehicle's perception. This paper presents an attack that uses light-emitting diodes and exploits the camera's rolling shutter effect to create adversarial stripes in the captured images to mislead traffic sign recognition. The attack is stealthy because the stripes on the traffic sign are invisible to human. For the attack to be threatening, the recognition results need to be stable over consecutive image frames. To achieve this, we design and implement GhostStripe, an attack system that controls the timing of the modulated light emission to adapt to camera operations and victim vehicle movements. Evaluated on real testbeds, GhostStripe can stably spoof the traffic sign recognition results for up to 94\% of frames to a wrong class when the victim vehicle passes the road section. In reality, such attack effect may fool victim vehicles into life-threatening incidents. We discuss the countermeasures at the levels of camera sensor, perception model, and autonomous driving system.



## **20. Formal Verification of Object Detection**

cs.CV

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.01295v3) [paper-pdf](http://arxiv.org/pdf/2407.01295v3)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.



## **21. A Survey of Attacks on Large Vision-Language Models: Resources, Advances, and Future Trends**

cs.CV

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07403v1) [paper-pdf](http://arxiv.org/pdf/2407.07403v1)

**Authors**: Daizong Liu, Mingyu Yang, Xiaoye Qu, Pan Zhou, Wei Hu, Yu Cheng

**Abstract**: With the significant development of large models in recent years, Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a wide range of multimodal understanding and reasoning tasks. Compared to traditional Large Language Models (LLMs), LVLMs present great potential and challenges due to its closer proximity to the multi-resource real-world applications and the complexity of multi-modal processing. However, the vulnerability of LVLMs is relatively underexplored, posing potential security risks in daily usage. In this paper, we provide a comprehensive review of the various forms of existing LVLM attacks. Specifically, we first introduce the background of attacks targeting LVLMs, including the attack preliminary, attack challenges, and attack resources. Then, we systematically review the development of LVLM attack methods, such as adversarial attacks that manipulate model outputs, jailbreak attacks that exploit model vulnerabilities for unauthorized actions, prompt injection attacks that engineer the prompt type and pattern, and data poisoning that affects model training. Finally, we discuss promising research directions in the future. We believe that our survey provides insights into the current landscape of LVLM vulnerabilities, inspiring more researchers to explore and mitigate potential safety issues in LVLM developments. The latest papers on LVLM attacks are continuously collected in https://github.com/liudaizong/Awesome-LVLM-Attack.



## **22. Marlin: Knowledge-Driven Analysis of Provenance Graphs for Efficient and Robust Detection of Cyber Attacks**

cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2403.12541v2) [paper-pdf](http://arxiv.org/pdf/2403.12541v2)

**Authors**: Zhenyuan Li, Yangyang Wei, Xiangmin Shen, Lingzhi Wang, Yan Chen, Haitao Xu, Shouling Ji, Fan Zhang, Liang Hou, Wenmao Liu, Xuhong Zhang, Jianwei Ying

**Abstract**: Recent research in both academia and industry has validated the effectiveness of provenance graph-based detection for advanced cyber attack detection and investigation. However, analyzing large-scale provenance graphs often results in substantial overhead. To improve performance, existing detection systems implement various optimization strategies. Yet, as several recent studies suggest, these strategies could lose necessary context information and be vulnerable to evasions. Designing a detection system that is efficient and robust against adversarial attacks is an open problem. We introduce Marlin, which approaches cyber attack detection through real-time provenance graph alignment.By leveraging query graphs embedded with attack knowledge, Marlin can efficiently identify entities and events within provenance graphs, embedding targeted analysis and significantly narrowing the search space. Moreover, we incorporate our graph alignment algorithm into a tag propagation-based schema to eliminate the need for storing and reprocessing raw logs. This design significantly reduces in-memory storage requirements and minimizes data processing overhead. As a result, it enables real-time graph alignment while preserving essential context information, thereby enhancing the robustness of cyber attack detection. Moreover, Marlin allows analysts to customize attack query graphs flexibly to detect extended attacks and provide interpretable detection results. We conduct experimental evaluations on two large-scale public datasets containing 257.42 GB of logs and 12 query graphs of varying sizes, covering multiple attack techniques and scenarios. The results show that Marlin can process 137K events per second while accurately identifying 120 subgraphs with 31 confirmed attacks, along with only 1 false positive, demonstrating its efficiency and accuracy in handling massive data.



## **23. Characterizing Encrypted Application Traffic through Cellular Radio Interface Protocol**

cs.NI

9 pages, 8 figures, 2 tables. This paper has been accepted for  publication by the 21st IEEE International Conference on Mobile Ad-Hoc and  Smart Systems (MASS 2024)

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07361v1) [paper-pdf](http://arxiv.org/pdf/2407.07361v1)

**Authors**: Md Ruman Islam, Raja Hasnain Anwar, Spyridon Mastorakis, Muhammad Taqi Raza

**Abstract**: Modern applications are end-to-end encrypted to prevent data from being read or secretly modified. 5G tech nology provides ubiquitous access to these applications without compromising the application-specific performance and latency goals. In this paper, we empirically demonstrate that 5G radio communication becomes the side channel to precisely infer the user's applications in real-time. The key idea lies in observing the 5G physical and MAC layer interactions over time that reveal the application's behavior. The MAC layer receives the data from the application and requests the network to assign the radio resource blocks. The network assigns the radio resources as per application requirements, such as priority, Quality of Service (QoS) needs, amount of data to be transmitted, and buffer size. The adversary can passively observe the radio resources to fingerprint the applications. We empirically demonstrate this attack by considering four different categories of applications: online shopping, voice/video conferencing, video streaming, and Over-The-Top (OTT) media platforms. Finally, we have also demonstrated that an attacker can differentiate various types of applications in real-time within each category.



## **24. The Quantum Imitation Game: Reverse Engineering of Quantum Machine Learning Models**

quant-ph

10 pages, 12 figures

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.07237v1) [paper-pdf](http://arxiv.org/pdf/2407.07237v1)

**Authors**: Archisman Ghosh, Swaroop Ghosh

**Abstract**: Quantum Machine Learning (QML) amalgamates quantum computing paradigms with machine learning models, providing significant prospects for solving complex problems. However, with the expansion of numerous third-party vendors in the Noisy Intermediate-Scale Quantum (NISQ) era of quantum computing, the security of QML models is of prime importance, particularly against reverse engineering, which could expose trained parameters and algorithms of the models. We assume the untrusted quantum cloud provider is an adversary having white-box access to the transpiled user-designed trained QML model during inference. Reverse engineering (RE) to extract the pre-transpiled QML circuit will enable re-transpilation and usage of the model for various hardware with completely different native gate sets and even different qubit technology. Such flexibility may not be obtained from the transpiled circuit which is tied to a particular hardware and qubit technology. The information about the number of parameters, and optimized values can allow further training of the QML model to alter the QML model, tamper with the watermark, and/or embed their own watermark or refine the model for other purposes. In this first effort to investigate the RE of QML circuits, we perform RE and compare the training accuracy of original and reverse-engineered Quantum Neural Networks (QNNs) of various sizes. We note that multi-qubit classifiers can be reverse-engineered under specific conditions with a mean error of order 1e-2 in a reasonable time. We also propose adding dummy fixed parametric gates in the QML models to increase the RE overhead for defense. For instance, adding 2 dummy qubits and 2 layers increases the overhead by ~1.76 times for a classifier with 2 qubits and 3 layers with a performance overhead of less than 9%. We note that RE is a very powerful attack model which warrants further efforts on defenses.



## **25. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

cs.IR

Survey paper

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06992v1) [paper-pdf](http://arxiv.org/pdf/2407.06992v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.



## **26. Does CLIP Know My Face?**

cs.LG

Published in the Journal of Artificial Intelligence Research (JAIR)

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2209.07341v4) [paper-pdf](http://arxiv.org/pdf/2209.07341v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data have become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.



## **27. Performance Evaluation of Knowledge Graph Embedding Approaches under Non-adversarial Attacks**

cs.LG

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06855v1) [paper-pdf](http://arxiv.org/pdf/2407.06855v1)

**Authors**: Sourabh Kapoor, Arnab Sharma, Michael Röder, Caglar Demir, Axel-Cyrille Ngonga Ngomo

**Abstract**: Knowledge Graph Embedding (KGE) transforms a discrete Knowledge Graph (KG) into a continuous vector space facilitating its use in various AI-driven applications like Semantic Search, Question Answering, or Recommenders. While KGE approaches are effective in these applications, most existing approaches assume that all information in the given KG is correct. This enables attackers to influence the output of these approaches, e.g., by perturbing the input. Consequently, the robustness of such KGE approaches has to be addressed. Recent work focused on adversarial attacks. However, non-adversarial attacks on all attack surfaces of these approaches have not been thoroughly examined. We close this gap by evaluating the impact of non-adversarial attacks on the performance of 5 state-of-the-art KGE algorithms on 5 datasets with respect to attacks on 3 attack surfaces-graph, parameter, and label perturbation. Our evaluation results suggest that label perturbation has a strong effect on the KGE performance, followed by parameter perturbation with a moderate and graph with a low effect.



## **28. EvolBA: Evolutionary Boundary Attack under Hard-label Black Box condition**

cs.CV

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.02248v3) [paper-pdf](http://arxiv.org/pdf/2407.02248v3)

**Authors**: Ayane Tajima, Satoshi Ono

**Abstract**: Research has shown that deep neural networks (DNNs) have vulnerabilities that can lead to the misrecognition of Adversarial Examples (AEs) with specifically designed perturbations. Various adversarial attack methods have been proposed to detect vulnerabilities under hard-label black box (HL-BB) conditions in the absence of loss gradients and confidence scores.However, these methods fall into local solutions because they search only local regions of the search space. Therefore, this study proposes an adversarial attack method named EvolBA to generate AEs using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) under the HL-BB condition, where only a class label predicted by the target DNN model is available. Inspired by formula-driven supervised learning, the proposed method introduces domain-independent operators for the initialization process and a jump that enhances search exploration. Experimental results confirmed that the proposed method could determine AEs with smaller perturbations than previous methods in images where the previous methods have difficulty.



## **29. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

cs.CR

Accepted to IEEE Euro S&P 2024

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2401.04929v3) [paper-pdf](http://arxiv.org/pdf/2401.04929v3)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.



## **30. A Hybrid Training-time and Run-time Defense Against Adversarial Attacks in Modulation Classification**

cs.AI

Published in IEEE Wireless Communications Letters, vol. 11, no. 6,  pp. 1161-1165, June 2022

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06807v1) [paper-pdf](http://arxiv.org/pdf/2407.06807v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Ambra Demontis, Fabio Roli

**Abstract**: Motivated by the superior performance of deep learning in many applications including computer vision and natural language processing, several recent studies have focused on applying deep neural network for devising future generations of wireless networks. However, several recent works have pointed out that imperceptible and carefully designed adversarial examples (attacks) can significantly deteriorate the classification accuracy. In this paper, we investigate a defense mechanism based on both training-time and run-time defense techniques for protecting machine learning-based radio signal (modulation) classification against adversarial attacks. The training-time defense consists of adversarial training and label smoothing, while the run-time defense employs a support vector machine-based neural rejection (NR). Considering a white-box scenario and real datasets, we demonstrate that our proposed techniques outperform existing state-of-the-art technologies.



## **31. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.08298v4) [paper-pdf](http://arxiv.org/pdf/2406.08298v4)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.



## **32. Countermeasures Against Adversarial Examples in Radio Signal Classification**

cs.AI

Published in IEEE Wireless Communications Letters, vol. 10, no. 8,  pp. 1830-1834, Aug. 2021

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06796v1) [paper-pdf](http://arxiv.org/pdf/2407.06796v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Basil AsSadhan, Fabio Roli

**Abstract**: Deep learning algorithms have been shown to be powerful in many communication network design problems, including that in automatic modulation classification. However, they are vulnerable to carefully crafted attacks called adversarial examples. Hence, the reliance of wireless networks on deep learning algorithms poses a serious threat to the security and operation of wireless networks. In this letter, we propose for the first time a countermeasure against adversarial examples in modulation classification. Our countermeasure is based on a neural rejection technique, augmented by label smoothing and Gaussian noise injection, that allows to detect and reject adversarial examples with high accuracy. Our results demonstrate that the proposed countermeasure can protect deep-learning based modulation classification systems against adversarial examples.



## **33. Diffusion-Based Adversarial Purification for Speaker Verification**

eess.AS

Accepted by IEEE Signal Processing Letters

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2310.14270v3) [paper-pdf](http://arxiv.org/pdf/2310.14270v3)

**Authors**: Yibo Bai, Xiao-Lei Zhang, Xuelong Li

**Abstract**: Recently, automatic speaker verification (ASV) based on deep learning is easily contaminated by adversarial attacks, which is a new type of attack that injects imperceptible perturbations to audio signals so as to make ASV produce wrong decisions. This poses a significant threat to the security and reliability of ASV systems. To address this issue, we propose a Diffusion-Based Adversarial Purification (DAP) method that enhances the robustness of ASV systems against such adversarial attacks. Our method leverages a conditional denoising diffusion probabilistic model to effectively purify the adversarial examples and mitigate the impact of perturbations. DAP first introduces controlled noise into adversarial examples, and then performs a reverse denoising process to reconstruct clean audio. Experimental results demonstrate the efficacy of the proposed DAP in enhancing the security of ASV and meanwhile minimizing the distortion of the purified audio signals.



## **34. Improving the Transferability of Adversarial Examples by Feature Augmentation**

cs.CV

19 pages, 4 figures, 4 tables

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06714v1) [paper-pdf](http://arxiv.org/pdf/2407.06714v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Xiaohu Zheng, Junqi Wu, Xiaoqian Chen

**Abstract**: Despite the success of input transformation-based attacks on boosting adversarial transferability, the performance is unsatisfying due to the ignorance of the discrepancy across models. In this paper, we propose a simple but effective feature augmentation attack (FAUG) method, which improves adversarial transferability without introducing extra computation costs. Specifically, we inject the random noise into the intermediate features of the model to enlarge the diversity of the attack gradient, thereby mitigating the risk of overfitting to the specific model and notably amplifying adversarial transferability. Moreover, our method can be combined with existing gradient attacks to augment their performance further. Extensive experiments conducted on the ImageNet dataset across CNN and transformer models corroborate the efficacy of our method, e.g., we achieve improvement of +26.22% and +5.57% on input transformation-based attacks and combination methods, respectively.



## **35. Universal Multi-view Black-box Attack against Object Detectors via Layout Optimization**

cs.CV

12 pages, 13 figures, 5 tables

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06688v1) [paper-pdf](http://arxiv.org/pdf/2407.06688v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Chao Li, Xiaoqian Chen

**Abstract**: Object detectors have demonstrated vulnerability to adversarial examples crafted by small perturbations that can deceive the object detector. Existing adversarial attacks mainly focus on white-box attacks and are merely valid at a specific viewpoint, while the universal multi-view black-box attack is less explored, limiting their generalization in practice. In this paper, we propose a novel universal multi-view black-box attack against object detectors, which optimizes a universal adversarial UV texture constructed by multiple image stickers for a 3D object via the designed layout optimization algorithm. Specifically, we treat the placement of image stickers on the UV texture as a circle-based layout optimization problem, whose objective is to find the optimal circle layout filled with image stickers so that it can deceive the object detector under the multi-view scenario. To ensure reasonable placement of image stickers, two constraints are elaborately devised. To optimize the layout, we adopt the random search algorithm enhanced by the devised important-aware selection strategy to find the most appropriate image sticker for each circle from the image sticker pools. Extensive experiments conducted on four common object detectors suggested that the detection performance decreases by a large magnitude of 74.29% on average in multi-view scenarios. Additionally, a novel evaluation tool based on the photo-realistic simulator is designed to assess the texture-based attack fairly.



## **36. Attack GAN (AGAN ): A new Security Evaluation Tool for Perceptual Encryption**

cs.CV

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06570v1) [paper-pdf](http://arxiv.org/pdf/2407.06570v1)

**Authors**: Umesh Kashyap, Sudev Kumar Padhi, Sk. Subidh Ali

**Abstract**: Training state-of-the-art (SOTA) deep learning models requires a large amount of data. The visual information present in the training data can be misused, which creates a huge privacy concern. One of the prominent solutions for this issue is perceptual encryption, which converts images into an unrecognizable format to protect the sensitive visual information in the training data. This comes at the cost of a significant reduction in the accuracy of the models. Adversarial Visual Information Hiding (AV IH) overcomes this drawback to protect image privacy by attempting to create encrypted images that are unrecognizable to the human eye while keeping relevant features for the target model. In this paper, we introduce the Attack GAN (AGAN ) method, a new Generative Adversarial Network (GAN )-based attack that exposes multiple vulnerabilities in the AV IH method. To show the adaptability, the AGAN is extended to traditional perceptual encryption methods of Learnable encryption (LE) and Encryption-then-Compression (EtC). Extensive experiments were conducted on diverse image datasets and target models to validate the efficacy of our AGAN method. The results show that AGAN can successfully break perceptual encryption methods by reconstructing original images from their AV IH encrypted images. AGAN can be used as a benchmark tool to evaluate the robustness of encryption methods for privacy protection such as AV IH.



## **37. DLOVE: A new Security Evaluation Tool for Deep Learning Based Watermarking Techniques**

cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06552v1) [paper-pdf](http://arxiv.org/pdf/2407.06552v1)

**Authors**: Sudev Kumar Padhi, Sk. Subidh Ali

**Abstract**: Recent developments in Deep Neural Network (DNN) based watermarking techniques have shown remarkable performance. The state-of-the-art DNN-based techniques not only surpass the robustness of classical watermarking techniques but also show their robustness against many image manipulation techniques. In this paper, we performed a detailed security analysis of different DNN-based watermarking techniques. We propose a new class of attack called the Deep Learning-based OVErwriting (DLOVE) attack, which leverages adversarial machine learning and overwrites the original embedded watermark with a targeted watermark in a watermarked image. To the best of our knowledge, this attack is the first of its kind. We have considered scenarios where watermarks are used to devise and formulate an adversarial attack in white box and black box settings. To show adaptability and efficiency, we launch our DLOVE attack analysis on seven different watermarking techniques, HiDDeN, ReDMark, PIMoG, Stegastamp, Aparecium, Distortion Agostic Deep Watermarking and Hiding Images in an Image. All these techniques use different approaches to create imperceptible watermarked images. Our attack analysis on these watermarking techniques with various constraints highlights the vulnerabilities of DNN-based watermarking. Extensive experimental results validate the capabilities of DLOVE. We propose DLOVE as a benchmark security analysis tool to test the robustness of future deep learning-based watermarking techniques.



## **38. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

cs.CL

First two authors contributed equally. Third and fourth authors  contributed equally

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.18495v2) [paper-pdf](http://arxiv.org/pdf/2406.18495v2)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.



## **39. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.03230v3) [paper-pdf](http://arxiv.org/pdf/2406.03230v3)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **40. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2401.17263v4) [paper-pdf](http://arxiv.org/pdf/2401.17263v4)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo



## **41. Non-Robust Features are Not Always Useful in One-Class Classification**

cs.LG

CVPR Visual and Anomaly Detection (VAND) Workshop 2024

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.06372v1) [paper-pdf](http://arxiv.org/pdf/2407.06372v1)

**Authors**: Matthew Lau, Haoran Wang, Alec Helbling, Matthew Hul, ShengYun Peng, Martin Andreoni, Willian T. Lunardi, Wenke Lee

**Abstract**: The robustness of machine learning models has been questioned by the existence of adversarial examples. We examine the threat of adversarial examples in practical applications that require lightweight models for one-class classification. Building on Ilyas et al. (2019), we investigate the vulnerability of lightweight one-class classifiers to adversarial attacks and possible reasons for it. Our results show that lightweight one-class classifiers learn features that are not robust (e.g. texture) under stronger attacks. However, unlike in multi-class classification (Ilyas et al., 2019), these non-robust features are not always useful for the one-class task, suggesting that learning these unpredictive and non-robust features is an unwanted consequence of training.



## **42. Improving Alignment and Robustness with Circuit Breakers**

cs.LG

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04313v3) [paper-pdf](http://arxiv.org/pdf/2406.04313v3)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.



## **43. Adaptive and robust watermark against model extraction attack**

cs.CR

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2405.02365v2) [paper-pdf](http://arxiv.org/pdf/2405.02365v2)

**Authors**: Kaiyi Pang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.



## **44. Multi-View Black-Box Physical Attacks on Infrared Pedestrian Detectors Using Adversarial Infrared Grid**

cs.CV

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.01168v2) [paper-pdf](http://arxiv.org/pdf/2407.01168v2)

**Authors**: Kalibinuer Tiliwalidi, Chengyin Hu, Weiwen Shi

**Abstract**: While extensive research exists on physical adversarial attacks within the visible spectrum, studies on such techniques in the infrared spectrum are limited. Infrared object detectors are vital in modern technological applications but are susceptible to adversarial attacks, posing significant security threats. Previous studies using physical perturbations like light bulb arrays and aerogels for white-box attacks, or hot and cold patches for black-box attacks, have proven impractical or limited in multi-view support. To address these issues, we propose the Adversarial Infrared Grid (AdvGrid), which models perturbations in a grid format and uses a genetic algorithm for black-box optimization. These perturbations are cyclically applied to various parts of a pedestrian's clothing to facilitate multi-view black-box physical attacks on infrared pedestrian detectors. Extensive experiments validate AdvGrid's effectiveness, stealthiness, and robustness. The method achieves attack success rates of 80.00\% in digital environments and 91.86\% in physical environments, outperforming baseline methods. Additionally, the average attack success rate exceeds 50\% against mainstream detectors, demonstrating AdvGrid's robustness. Our analyses include ablation studies, transfer attacks, and adversarial defenses, confirming the method's superiority.



## **45. Malicious Agent Detection for Robust Multi-Agent Collaborative Perception**

cs.CR

Accepted by IROS 2024

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2310.11901v2) [paper-pdf](http://arxiv.org/pdf/2310.11901v2)

**Authors**: Yangheng Zhao, Zhen Xiang, Sheng Yin, Xianghe Pang, Siheng Chen, Yanfeng Wang

**Abstract**: Recently, multi-agent collaborative (MAC) perception has been proposed and outperformed the traditional single-agent perception in many applications, such as autonomous driving. However, MAC perception is more vulnerable to adversarial attacks than single-agent perception due to the information exchange. The attacker can easily degrade the performance of a victim agent by sending harmful information from a malicious agent nearby. In this paper, we extend adversarial attacks to an important perception task -- MAC object detection, where generic defenses such as adversarial training are no longer effective against these attacks. More importantly, we propose Malicious Agent Detection (MADE), a reactive defense specific to MAC perception that can be deployed by each agent to accurately detect and then remove any potential malicious agent in its local collaboration network. In particular, MADE inspects each agent in the network independently using a semi-supervised anomaly detector based on a double-hypothesis test with the Benjamini-Hochberg procedure to control the false positive rate of the inference. For the two hypothesis tests, we propose a match loss statistic and a collaborative reconstruction loss statistic, respectively, both based on the consistency between the agent to be inspected and the ego agent where our detector is deployed. We conduct comprehensive evaluations on a benchmark 3D dataset V2X-sim and a real-road dataset DAIR-V2X and show that with the protection of MADE, the drops in the average precision compared with the best-case "oracle" defender against our attack are merely 1.28% and 0.34%, respectively, much lower than 8.92% and 10.00% for adversarial training, respectively.



## **46. Exploring the Adversarial Capabilities of Large Language Models**

cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2402.09132v4) [paper-pdf](http://arxiv.org/pdf/2402.09132v4)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.



## **47. Improving Adversarial Transferability of Vision-Language Pre-training Models through Collaborative Multimodal Interaction**

cs.CV

This work won first place in CVPR 2024 Workshop Challenge: Black-box  Adversarial Attacks on Vision Foundation Models

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2403.10883v2) [paper-pdf](http://arxiv.org/pdf/2403.10883v2)

**Authors**: Jiyuan Fu, Zhaoyu Chen, Kaixun Jiang, Haijing Guo, Jiafeng Wang, Shuyong Gao, Wenqiang Zhang

**Abstract**: Despite the substantial advancements in Vision-Language Pre-training (VLP) models, their susceptibility to adversarial attacks poses a significant challenge. Existing work rarely studies the transferability of attacks on VLP models, resulting in a substantial performance gap from white-box attacks. We observe that prior work overlooks the interaction mechanisms between modalities, which plays a crucial role in understanding the intricacies of VLP models. In response, we propose a novel attack, called Collaborative Multimodal Interaction Attack (CMI-Attack), leveraging modality interaction through embedding guidance and interaction enhancement. Specifically, attacking text at the embedding level while preserving semantics, as well as utilizing interaction image gradients to enhance constraints on perturbations of texts and images. Significantly, in the image-text retrieval task on Flickr30K dataset, CMI-Attack raises the transfer success rates from ALBEF to TCL, $\text{CLIP}_{\text{ViT}}$ and $\text{CLIP}_{\text{CNN}}$ by 8.11%-16.75% over state-of-the-art methods. Moreover, CMI-Attack also demonstrates superior performance in cross-task generalization scenarios. Our work addresses the underexplored realm of transfer attacks on VLP models, shedding light on the importance of modality interaction for enhanced adversarial robustness.



## **48. A Survey of Fragile Model Watermarking**

cs.CR

Submitted Signal Processing

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04809v4) [paper-pdf](http://arxiv.org/pdf/2406.04809v4)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.



## **49. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

cs.CV

Accepted by ECCV'24. Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2310.11868v4) [paper-pdf](http://arxiv.org/pdf/2310.11868v4)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models. Through extensive benchmarking, we evaluate the robustness of widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safetydriven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: There exist AI generations that may be offensive in nature.



## **50. Rethinking Targeted Adversarial Attacks For Neural Machine Translation**

cs.CL

5 pages, 2 figures, accepted by ICASSP 2024

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2407.05319v1) [paper-pdf](http://arxiv.org/pdf/2407.05319v1)

**Authors**: Junjie Wu, Lemao Liu, Wei Bi, Dit-Yan Yeung

**Abstract**: Targeted adversarial attacks are widely used to evaluate the robustness of neural machine translation systems. Unfortunately, this paper first identifies a critical issue in the existing settings of NMT targeted adversarial attacks, where their attacking results are largely overestimated. To this end, this paper presents a new setting for NMT targeted adversarial attacks that could lead to reliable attacking results. Under the new setting, it then proposes a Targeted Word Gradient adversarial Attack (TWGA) method to craft adversarial examples. Experimental results demonstrate that our proposed setting could provide faithful attacking results for targeted adversarial attacks on NMT systems, and the proposed TWGA method can effectively attack such victim NMT systems. In-depth analyses on a large-scale dataset further illustrate some valuable findings. 1 Our code and data are available at https://github.com/wujunjie1998/TWGA.



