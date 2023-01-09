# Latest Adversarial Attack Papers
**update at 2023-01-09 10:13:54**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Stealthy Backdoor Attack for Code Models**

cs.CR

Under review of IEEE Transactions on Software Engineering

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02496v1) [paper-pdf](http://arxiv.org/pdf/2301.02496v1)

**Authors**: Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo

**Abstract**: Code models, such as CodeBERT and CodeT5, offer general-purpose representations of code and play a vital role in supporting downstream automated software engineering tasks. Most recently, code models were revealed to be vulnerable to backdoor attacks. A code model that is backdoor-attacked can behave normally on clean examples but will produce pre-defined malicious outputs on examples injected with triggers that activate the backdoors. Existing backdoor attacks on code models use unstealthy and easy-to-detect triggers. This paper aims to investigate the vulnerability of code models with stealthy backdoor attacks. To this end, we propose AFRAIDOOR (Adversarial Feature as Adaptive Backdoor). AFRAIDOOR achieves stealthiness by leveraging adversarial perturbations to inject adaptive triggers into different inputs. We evaluate AFRAIDOOR on three widely adopted code models (CodeBERT, PLBART and CodeT5) and two downstream tasks (code summarization and method name prediction). We find that around 85% of adaptive triggers in AFRAIDOOR bypass the detection in the defense process. By contrast, only less than 12% of the triggers from previous work bypass the defense. When the defense method is not applied, both AFRAIDOOR and baselines have almost perfect attack success rates. However, once a defense is applied, the success rates of baselines decrease dramatically to 10.47% and 12.06%, while the success rate of AFRAIDOOR are 77.05% and 92.98% on the two tasks. Our finding exposes security weaknesses in code models under stealthy backdoor attacks and shows that the state-of-the-art defense method cannot provide sufficient protection. We call for more research efforts in understanding security threats to code models and developing more effective countermeasures.



## **2. Watching your call: Breaking VoLTE Privacy in LTE/5G Networks**

cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02487v1) [paper-pdf](http://arxiv.org/pdf/2301.02487v1)

**Authors**: Zishuai Cheng, Mihai Ordean, Flavio D. Garcia, Baojiang Cui, Dominik Rys

**Abstract**: Voice over LTE (VoLTE) and Voice over NR (VoNR) are two similar technologies that have been widely deployed by operators to provide a better calling experience in LTE and 5G networks, respectively. The VoLTE/NR protocols rely on the security features of the underlying LTE/5G network to protect users' privacy such that nobody can monitor calls and learn details about call times, duration, and direction. In this paper, we introduce a new privacy attack which enables adversaries to analyse encrypted LTE/5G traffic and recover any VoLTE/NR call details. We achieve this by implementing a novel mobile-relay adversary which is able to remain undetected by using an improved physical layer parameter guessing procedure. This adversary facilitates the recovery of encrypted configuration messages exchanged between victim devices and the mobile network. We further propose an identity mapping method which enables our mobile-relay adversary to link a victim's network identifiers to the phone number efficiently, requiring a single VoLTE protocol message. We evaluate the real-world performance of our attacks using four modern commercial off-the-shelf phones and two representative, commercial network carriers. We collect over 60 hours of traffic between the phones and the mobile networks and execute 160 VoLTE calls, which we use to successfully identify patterns in the physical layer parameter allocation and in VoLTE traffic, respectively. Our real-world experiments show that our mobile-relay works as expected in all test cases, and the VoLTE activity logs recovered describe the actual communication with 100% accuracy. Finally, we show that we can link network identifiers such as International Mobile Subscriber Identities (IMSI), Subscriber Concealed Identifiers (SUCI) and/or Globally Unique Temporary Identifiers (GUTI) to phone numbers while remaining undetected by the victim.



## **3. Adversarial Attacks on Neural Models of Code via Code Difference Reduction**

cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02412v1) [paper-pdf](http://arxiv.org/pdf/2301.02412v1)

**Authors**: Zhao Tian, Junjie Chen, Zhi Jin

**Abstract**: Deep learning has been widely used to solve various code-based tasks by building deep code models based on a large number of code snippets. However, deep code models are still vulnerable to adversarial attacks. As source code is discrete and has to strictly stick to the grammar and semantics constraints, the adversarial attack techniques in other domains are not applicable. Moreover, the attack techniques specific to deep code models suffer from the effectiveness issue due to the enormous attack space. In this work, we propose a novel adversarial attack technique (i.e., CODA). Its key idea is to use the code differences between the target input and reference inputs (that have small code differences but different prediction results with the target one) to guide the generation of adversarial examples. It considers both structure differences and identifier differences to preserve the original semantics. Hence, the attack space can be largely reduced as the one constituted by the two kinds of code differences, and thus the attack process can be largely improved by designing corresponding equivalent structure transformations and identifier renaming transformations. Our experiments on 10 deep code models (i.e., two pre trained models with five code-based tasks) demonstrate the effectiveness and efficiency of CODA, the naturalness of its generated examples, and its capability of defending against attacks after adversarial fine-tuning. For example, CODA improves the state-of-the-art techniques (i.e., CARROT and ALERT) by 79.25% and 72.20% on average in terms of the attack success rate, respectively.



## **4. TrojanPuzzle: Covertly Poisoning Code-Suggestion Models**

cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02344v1) [paper-pdf](http://arxiv.org/pdf/2301.02344v1)

**Authors**: Hojjat Aghakhani, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, Robert Sim

**Abstract**: With tools like GitHub Copilot, automatic code suggestion is no longer a dream in software engineering. These tools, based on large language models, are typically trained on massive corpora of code mined from unvetted public sources. As a result, these models are susceptible to data poisoning attacks where an adversary manipulates the model's training or fine-tuning phases by injecting malicious data. Poisoning attacks could be designed to influence the model's suggestions at run time for chosen contexts, such as inducing the model into suggesting insecure code payloads. To achieve this, prior poisoning attacks explicitly inject the insecure code payload into the training data, making the poisoning data detectable by static analysis tools that can remove such malicious data from the training set. In this work, we demonstrate two novel data poisoning attacks, COVERT and TROJANPUZZLE, that can bypass static analysis by planting malicious poisoning data in out-of-context regions such as docstrings. Our most novel attack, TROJANPUZZLE, goes one step further in generating less suspicious poisoning data by never including certain (suspicious) parts of the payload in the poisoned data, while still inducing a model that suggests the entire payload when completing code (i.e., outside docstrings). This makes TROJANPUZZLE robust against signature-based dataset-cleansing methods that identify and filter out suspicious sequences from the training data. Our evaluation against two model sizes demonstrates that both COVERT and TROJANPUZZLE have significant implications for how practitioners should select code used to train or tune code-suggestion models.



## **5. Silent Killer: Optimizing Backdoor Trigger Yields a Stealthy and Powerful Data Poisoning Attack**

cs.CR

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.02615v1) [paper-pdf](http://arxiv.org/pdf/2301.02615v1)

**Authors**: Tzvi Lederer, Gallil Maimon, Lior Rokach

**Abstract**: We propose a stealthy and powerful backdoor attack on neural networks based on data poisoning (DP). In contrast to previous attacks, both the poison and the trigger in our method are stealthy. We are able to change the model's classification of samples from a source class to a target class chosen by the attacker. We do so by using a small number of poisoned training samples with nearly imperceptible perturbations, without changing their labels. At inference time, we use a stealthy perturbation added to the attacked samples as a trigger. This perturbation is crafted as a universal adversarial perturbation (UAP), and the poison is crafted using gradient alignment coupled to this trigger. Our method is highly efficient in crafting time compared to previous methods and requires only a trained surrogate model without additional retraining. Our attack achieves state-of-the-art results in terms of attack success rate while maintaining high accuracy on clean samples.



## **6. Holistic Adversarial Robustness of Deep Learning Models**

cs.LG

survey paper on holistic adversarial robustness for deep learning;  published at AAAI 2023 Senior Member Presentation Track

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2202.07201v3) [paper-pdf](http://arxiv.org/pdf/2202.07201v3)

**Authors**: Pin-Yu Chen, Sijia Liu

**Abstract**: Adversarial robustness studies the worst-case performance of a machine learning model to ensure safety and reliability. With the proliferation of deep-learning-based technology, the potential risks associated with model development and deployment can be amplified and become dreadful vulnerabilities. This paper provides a comprehensive overview of research topics and foundational principles of research methods for adversarial robustness of deep learning models, including attacks, defenses, verification, and novel applications.



## **7. Randomized Message-Interception Smoothing: Gray-box Certificates for Graph Neural Networks**

cs.LG

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.02039v1) [paper-pdf](http://arxiv.org/pdf/2301.02039v1)

**Authors**: Yan Scholten, Jan Schuchardt, Simon Geisler, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Randomized smoothing is one of the most promising frameworks for certifying the adversarial robustness of machine learning models, including Graph Neural Networks (GNNs). Yet, existing randomized smoothing certificates for GNNs are overly pessimistic since they treat the model as a black box, ignoring the underlying architecture. To remedy this, we propose novel gray-box certificates that exploit the message-passing principle of GNNs: We randomly intercept messages and carefully analyze the probability that messages from adversarially controlled nodes reach their target nodes. Compared to existing certificates, we certify robustness to much stronger adversaries that control entire nodes in the graph and can arbitrarily manipulate node features. Our certificates provide stronger guarantees for attacks at larger distances, as messages from farther-away nodes are more likely to get intercepted. We demonstrate the effectiveness of our method on various models and datasets. Since our gray-box certificates consider the underlying graph structure, we can significantly improve certifiable robustness by applying graph sparsification.



## **8. Beckman Defense**

cs.LG

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.01495v2) [paper-pdf](http://arxiv.org/pdf/2301.01495v2)

**Authors**: A. V. Subramanyam

**Abstract**: Optimal transport (OT) based distributional robust optimisation (DRO) has received some traction in the recent past. However, it is at a nascent stage but has a sound potential in robustifying the deep learning models. Interestingly, OT barycenters demonstrate a good robustness against adversarial attacks. Owing to the computationally expensive nature of OT barycenters, they have not been investigated under DRO framework. In this work, we propose a new barycenter, namely Beckman barycenter, which can be computed efficiently and used for training the network to defend against adversarial attacks in conjunction with adversarial training. We propose a novel formulation of Beckman barycenter and analytically obtain the barycenter using the marginals of the input image. We show that the Beckman barycenter can be used to train adversarially trained networks to improve the robustness. Our training is extremely efficient as it requires only a single epoch of training. Elaborate experiments on CIFAR-10, CIFAR-100 and Tiny ImageNet demonstrate that training an adversarially robust network with Beckman barycenter can significantly increase the performance. Under auto attack, we get a a maximum boost of 10\% in CIFAR-10, 8.34\% in CIFAR-100 and 11.51\% in Tiny ImageNet. Our code is available at https://github.com/Visual-Conception-Group/test-barycentric-defense.



## **9. Enhancement attacks in biomedical machine learning**

stat.ML

13 pages, 3 figures

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.01885v1) [paper-pdf](http://arxiv.org/pdf/2301.01885v1)

**Authors**: Matthew Rosenblatt, Javid Dadashkarimi, Dustin Scheinost

**Abstract**: The prevalence of machine learning in biomedical research is rapidly growing, yet the trustworthiness of such research is often overlooked. While some previous works have investigated the ability of adversarial attacks to degrade model performance in medical imaging, the ability to falsely improve performance via recently-developed "enhancement attacks" may be a greater threat to biomedical machine learning. In the spirit of developing attacks to better understand trustworthiness, we developed three techniques to drastically enhance prediction performance of classifiers with minimal changes to features, including the enhancement of 1) within-dataset predictions, 2) a particular method over another, and 3) cross-dataset generalization. Our within-dataset enhancement framework falsely improved classifiers' accuracy from 50% to almost 100% while maintaining high feature similarities between original and enhanced data (Pearson's r's>0.99). Similarly, the method-specific enhancement framework was effective in falsely improving the performance of one method over another. For example, a simple neural network outperformed LR by 50% on our enhanced dataset, although no performance differences were present in the original dataset. Crucially, the original and enhanced data were still similar (r=0.95). Finally, we demonstrated that enhancement is not specific to within-dataset predictions but can also be adapted to enhance the generalization accuracy of one dataset to another by up to 38%. Overall, our results suggest that more robust data sharing and provenance tracking pipelines are necessary to maintain data integrity in biomedical machine learning research.



## **10. Availability Adversarial Attack and Countermeasures for Deep Learning-based Load Forecasting**

cs.LG

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01832v1) [paper-pdf](http://arxiv.org/pdf/2301.01832v1)

**Authors**: Wangkun Xu, Fei Teng

**Abstract**: The forecast of electrical loads is essential for the planning and operation of the power system. Recently, advances in deep learning have enabled more accurate forecasts. However, deep neural networks are prone to adversarial attacks. Although most of the literature focuses on integrity-based attacks, this paper proposes availability-based adversarial attacks, which can be more easily implemented by attackers. For each forecast instance, the availability attack position is optimally solved by mixed-integer reformulation of the artificial neural network. To tackle this attack, an adversarial training algorithm is proposed. In simulation, a realistic load forecasting dataset is considered and the attack performance is compared to the integrity-based attack. Meanwhile, the adversarial training algorithm is shown to significantly improve robustness against availability attacks. All codes are available at https://github.com/xuwkk/AAA_Load_Forecast.



## **11. GUAP: Graph Universal Attack Through Adversarial Patching**

cs.LG

8 pages

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01731v1) [paper-pdf](http://arxiv.org/pdf/2301.01731v1)

**Authors**: Xiao Zang, Jie Chen, Bo Yuan

**Abstract**: Graph neural networks (GNNs) are a class of effective deep learning models for node classification tasks; yet their predictive capability may be severely compromised under adversarially designed unnoticeable perturbations to the graph structure and/or node data. Most of the current work on graph adversarial attacks aims at lowering the overall prediction accuracy, but we argue that the resulting abnormal model performance may catch attention easily and invite quick counterattack. Moreover, attacks through modification of existing graph data may be hard to conduct if good security protocols are implemented. In this work, we consider an easier attack harder to be noticed, through adversarially patching the graph with new nodes and edges. The attack is universal: it targets a single node each time and flips its connection to the same set of patch nodes. The attack is unnoticeable: it does not modify the predictions of nodes other than the target. We develop an algorithm, named GUAP, that achieves high attack success rate but meanwhile preserves the prediction accuracy. GUAP is fast to train by employing a sampling strategy. We demonstrate that a 5% sampling in each epoch yields 20x speedup in training, with only a slight degradation in attack performance. Additionally, we show that the adversarial patch trained with the graph convolutional network transfers well to other GNNs, such as the graph attention network.



## **12. A Survey on Physical Adversarial Attack in Computer Vision**

cs.CV

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2209.14262v2) [paper-pdf](http://arxiv.org/pdf/2209.14262v2)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Guijian Tang, Xiaoqian Chen

**Abstract**: In the past decade, deep learning has dramatically changed the traditional hand-craft feature manner with strong feature learning capability, resulting in tremendous improvement of conventional tasks. However, deep neural networks have recently been demonstrated vulnerable to adversarial examples, a kind of malicious samples crafted by small elaborately designed noise, which mislead the DNNs to make the wrong decisions while remaining imperceptible to humans. Adversarial examples can be divided into digital adversarial attacks and physical adversarial attacks. The digital adversarial attack is mostly performed in lab environments, focusing on improving the performance of adversarial attack algorithms. In contrast, the physical adversarial attack focus on attacking the physical world deployed DNN systems, which is a more challenging task due to the complex physical environment (i.e., brightness, occlusion, and so on). Although the discrepancy between digital adversarial and physical adversarial examples is small, the physical adversarial examples have a specific design to overcome the effect of the complex physical environment. In this paper, we review the development of physical adversarial attacks in DNN-based computer vision tasks, including image recognition tasks, object detection tasks, and semantic segmentation. For the sake of completeness of the algorithm evolution, we will briefly introduce the works that do not involve the physical adversarial attack. We first present a categorization scheme to summarize the current physical adversarial attacks. Then discuss the advantages and disadvantages of the existing physical adversarial attacks and focus on the technique used to maintain the adversarial when applied into physical environment. Finally, we point out the issues of the current physical adversarial attacks to be solved and provide promising research directions.



## **13. Validity in Music Information Research Experiments**

cs.SD

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01578v1) [paper-pdf](http://arxiv.org/pdf/2301.01578v1)

**Authors**: Bob L. T. Sturm, Arthur Flexer

**Abstract**: Validity is the truth of an inference made from evidence, such as data collected in an experiment, and is central to working scientifically. Given the maturity of the domain of music information research (MIR), validity in our opinion should be discussed and considered much more than it has been so far. Considering validity in one's work can improve its scientific and engineering value. Puzzling MIR phenomena like adversarial attacks and performance glass ceilings become less mysterious through the lens of validity. In this article, we review the subject of validity in general, considering the four major types of validity from a key reference: Shadish et al. 2002. We ground our discussion of these types with a prototypical MIR experiment: music classification using machine learning. Through this MIR experimentalists can be guided to make valid inferences from data collected from their experiments.



## **14. Passive Triangulation Attack on ORide**

cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2208.12216v3) [paper-pdf](http://arxiv.org/pdf/2208.12216v3)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.



## **15. The Feasibility and Inevitability of Stealth Attacks**

cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2106.13997v4) [paper-pdf](http://arxiv.org/pdf/2106.13997v4)

**Authors**: Ivan Y. Tyukin, Desmond J. Higham, Alexander Bastounis, Eliyas Woldegeorgis, Alexander N. Gorban

**Abstract**: We develop and study new adversarial perturbations that enable an attacker to gain control over decisions in generic Artificial Intelligence (AI) systems including deep learning neural networks. In contrast to adversarial data modification, the attack mechanism we consider here involves alterations to the AI system itself. Such a stealth attack could be conducted by a mischievous, corrupt or disgruntled member of a software development team. It could also be made by those wishing to exploit a ``democratization of AI'' agenda, where network architectures and trained parameter sets are shared publicly. We develop a range of new implementable attack strategies with accompanying analysis, showing that with high probability a stealth attack can be made transparent, in the sense that system performance is unchanged on a fixed validation set which is unknown to the attacker, while evoking any desired output on a trigger input of interest. The attacker only needs to have estimates of the size of the validation set and the spread of the AI's relevant latent space. In the case of deep learning neural networks, we show that a one neuron attack is possible - a modification to the weights and bias associated with a single neuron - revealing a vulnerability arising from over-parameterization. We illustrate these concepts using state of the art architectures on two standard image data sets. Guided by the theory and computational results, we also propose strategies to guard against stealth attacks.



## **16. Driver Locations Harvesting Attack on pRide**

cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2210.13263v3) [paper-pdf](http://arxiv.org/pdf/2210.13263v3)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.



## **17. Organised Firestorm as strategy for business cyber-attacks**

cs.CY

9 pages, 3 figures, 2 table

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01518v1) [paper-pdf](http://arxiv.org/pdf/2301.01518v1)

**Authors**: Andrea Russo

**Abstract**: Having a good reputation is paramount for most organisations and companies. In fact, having an optimal corporate image allows them to have better transaction relationships with various customers and partners. However, such reputation is hard to build and easy to destroy for all kind of business commercial activities (B2C, B2B, B2B2C, B2G). A misunderstanding during the communication process to the customers, or just a bad communication strategy, can lead to a disaster for the entire company. This is emphasised by the reaction of millions of people on social networks, which can be very detrimental for the corporate image if they react negatively to a certain event. This is called a firestorm.   In this paper, I propose a well-organised strategy for firestorm attacks on organisations, also showing how an adversary can leverage them to obtain private information on the attacked firm. Standard business security procedures are not designed to operate against multi-domain attacks; therefore, I will show how it is possible to bypass the classic and advised security procedures by operating different kinds of attack. I also propose a different firestorm attack, targeting a specific business company network in an efficient way. Finally, I present defensive procedures to reduce the negative effect of firestorms on a company.



## **18. Universal adversarial perturbation for remote sensing images**

cs.CV

Published in the Twenty-Fourth International Workshop on Multimedia  Signal Processing, MMSP 2022

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2202.10693v2) [paper-pdf](http://arxiv.org/pdf/2202.10693v2)

**Authors**: Qingyu Wang, Guorui Feng, Zhaoxia Yin, Bin Luo

**Abstract**: Recently, with the application of deep learning in the remote sensing image (RSI) field, the classification accuracy of the RSI has been dramatically improved compared with traditional technology. However, even the state-of-the-art object recognition convolutional neural networks are fooled by the universal adversarial perturbation (UAP). The research on UAP is mostly limited to ordinary images, and RSIs have not been studied. To explore the basic characteristics of UAPs of RSIs, this paper proposes a novel method combining an encoder-decoder network with an attention mechanism to generate the UAP of RSIs. Firstly, the former is used to generate the UAP, which can learn the distribution of perturbations better, and then the latter is used to find the sensitive regions concerned by the RSI classification model. Finally, the generated regions are used to fine-tune the perturbation making the model misclassified with fewer perturbations. The experimental results show that the UAP can make the classification model misclassify, and the attack success rate of our proposed method on the RSI data set is as high as 97.09%.



## **19. Surveillance Face Anti-spoofing**

cs.CV

15 pages, 9 figures

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2301.00975v1) [paper-pdf](http://arxiv.org/pdf/2301.00975v1)

**Authors**: Hao Fang, Ajian Liu, Jun Wan, Sergio Escalera, Chenxu Zhao, Xu Zhang, Stan Z. Li, Zhen Lei

**Abstract**: Face Anti-spoofing (FAS) is essential to secure face recognition systems from various physical attacks. However, recent research generally focuses on short-distance applications (i.e., phone unlocking) while lacking consideration of long-distance scenes (i.e., surveillance security checks). In order to promote relevant research and fill this gap in the community, we collect a large-scale Surveillance High-Fidelity Mask (SuHiFiMask) dataset captured under 40 surveillance scenes, which has 101 subjects from different age groups with 232 3D attacks (high-fidelity masks), 200 2D attacks (posters, portraits, and screens), and 2 adversarial attacks. In this scene, low image resolution and noise interference are new challenges faced in surveillance FAS. Together with the SuHiFiMask dataset, we propose a Contrastive Quality-Invariance Learning (CQIL) network to alleviate the performance degradation caused by image quality from three aspects: (1) An Image Quality Variable module (IQV) is introduced to recover image information associated with discrimination by combining the super-resolution network. (2) Using generated sample pairs to simulate quality variance distributions to help contrastive learning strategies obtain robust feature representation under quality variation. (3) A Separate Quality Network (SQN) is designed to learn discriminative features independent of image quality. Finally, a large number of experiments verify the quality of the SuHiFiMask dataset and the superiority of the proposed CQIL.



## **20. Efficient Robustness Assessment via Adversarial Spatial-Temporal Focus on Videos**

cs.CV

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2301.00896v1) [paper-pdf](http://arxiv.org/pdf/2301.00896v1)

**Authors**: Wei Xingxing, Wang Songping, Yan Huanqian

**Abstract**: Adversarial robustness assessment for video recognition models has raised concerns owing to their wide applications on safety-critical tasks. Compared with images, videos have much high dimension, which brings huge computational costs when generating adversarial videos. This is especially serious for the query-based black-box attacks where gradient estimation for the threat models is usually utilized, and high dimensions will lead to a large number of queries. To mitigate this issue, we propose to simultaneously eliminate the temporal and spatial redundancy within the video to achieve an effective and efficient gradient estimation on the reduced searching space, and thus query number could decrease. To implement this idea, we design the novel Adversarial spatial-temporal Focus (AstFocus) attack on videos, which performs attacks on the simultaneously focused key frames and key regions from the inter-frames and intra-frames in the video. AstFocus attack is based on the cooperative Multi-Agent Reinforcement Learning (MARL) framework. One agent is responsible for selecting key frames, and another agent is responsible for selecting key regions. These two agents are jointly trained by the common rewards received from the black-box threat models to perform a cooperative prediction. By continuously querying, the reduced searching space composed of key frames and key regions is becoming precise, and the whole query number becomes less than that on the original video. Extensive experiments on four mainstream video recognition models and three widely used action recognition datasets demonstrate that the proposed AstFocus attack outperforms the SOTA methods, which is prevenient in fooling rate, query number, time, and perturbation magnitude at the same.



## **21. Adaptive Perturbation for Adversarial Attack**

cs.CV

13 pages, 5 figures, 9 tables

**SubmitDate**: 2023-01-02    [abs](http://arxiv.org/abs/2111.13841v2) [paper-pdf](http://arxiv.org/pdf/2111.13841v2)

**Authors**: Zheng Yuan, Jie Zhang, Zhaoyan Jiang, Liangliang Li, Shiguang Shan

**Abstract**: In recent years, the security of deep learning models achieves more and more attentions with the rapid development of neural networks, which are vulnerable to adversarial examples. Almost all existing gradient-based attack methods use the sign function in the generation to meet the requirement of perturbation budget on $L_\infty$ norm. However, we find that the sign function may be improper for generating adversarial examples since it modifies the exact gradient direction. Instead of using the sign function, we propose to directly utilize the exact gradient direction with a scaling factor for generating adversarial perturbations, which improves the attack success rates of adversarial examples even with fewer perturbations. At the same time, we also theoretically prove that this method can achieve better black-box transferability. Moreover, considering that the best scaling factor varies across different images, we propose an adaptive scaling factor generator to seek an appropriate scaling factor for each image, which avoids the computational cost for manually searching the scaling factor. Our method can be integrated with almost all existing gradient-based attack methods to further improve their attack success rates. Extensive experiments on the CIFAR10 and ImageNet datasets show that our method exhibits higher transferability and outperforms the state-of-the-art methods.



## **22. Differentiable Search of Accurate and Robust Architectures**

cs.LG

**SubmitDate**: 2023-01-02    [abs](http://arxiv.org/abs/2212.14049v2) [paper-pdf](http://arxiv.org/pdf/2212.14049v2)

**Authors**: Yuwei Ou, Xiangning Xie, Shangce Gao, Yanan Sun, Kay Chen Tan, Jiancheng Lv

**Abstract**: Deep neural networks (DNNs) are found to be vulnerable to adversarial attacks, and various methods have been proposed for the defense. Among these methods, adversarial training has been drawing increasing attention because of its simplicity and effectiveness. However, the performance of the adversarial training is greatly limited by the architectures of target DNNs, which often makes the resulting DNNs with poor accuracy and unsatisfactory robustness. To address this problem, we propose DSARA to automatically search for the neural architectures that are accurate and robust after adversarial training. In particular, we design a novel cell-based search space specially for adversarial training, which improves the accuracy and the robustness upper bound of the searched architectures by carefully designing the placement of the cells and the proportional relationship of the filter numbers. Then we propose a two-stage search strategy to search for both accurate and robust neural architectures. At the first stage, the architecture parameters are optimized to minimize the adversarial loss, which makes full use of the effectiveness of the adversarial training in enhancing the robustness. At the second stage, the architecture parameters are optimized to minimize both the natural loss and the adversarial loss utilizing the proposed multi-objective adversarial training method, so that the searched neural architectures are both accurate and robust. We evaluate the proposed algorithm under natural data and various adversarial attacks, which reveals the superiority of the proposed method in terms of both accurate and robust architectures. We also conclude that accurate and robust neural architectures tend to deploy very different structures near the input and the output, which has great practical significance on both hand-crafting and automatically designing of accurate and robust neural architectures.



## **23. Reversible Attack based on Local Visual Adversarial Perturbation**

cs.CV

**SubmitDate**: 2023-01-02    [abs](http://arxiv.org/abs/2110.02700v3) [paper-pdf](http://arxiv.org/pdf/2110.02700v3)

**Authors**: Li Chen, Shaowei Zhu, Zhaoxia Yin

**Abstract**: Adding perturbations to images can mislead classification models to produce incorrect results. Recently, researchers exploited adversarial perturbations to protect image privacy from retrieval by intelligent models. However, adding adversarial perturbations to images destroys the original data, making images useless in digital forensics and other fields. To prevent illegal or unauthorized access to sensitive image data such as human faces without impeding legitimate users, the use of reversible adversarial attack techniques is increasing. The original image can be recovered from its reversible adversarial examples. However, existing reversible adversarial attack methods are designed for traditional imperceptible adversarial perturbations and ignore the local visible adversarial perturbation. In this paper, we propose a new method for generating reversible adversarial examples based on local visible adversarial perturbation. The information needed for image recovery is embedded into the area beyond the adversarial patch by the reversible data hiding technique. To reduce image distortion, lossless compression and the B-R-G (bluered-green) embedding principle are adopted. Experiments on CIFAR-10 and ImageNet datasets show that the proposed method can restore the original images error-free while ensuring good attack performance.



## **24. Trojaning semi-supervised learning model via poisoning wild images on the web**

cs.CY

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2301.00435v1) [paper-pdf](http://arxiv.org/pdf/2301.00435v1)

**Authors**: Le Feng, Zhenxing Qian, Sheng Li, Xinpeng Zhang

**Abstract**: Wild images on the web are vulnerable to backdoor (also called trojan) poisoning, causing machine learning models learned on these images to be injected with backdoors. Most previous attacks assumed that the wild images are labeled. In reality, however, most images on the web are unlabeled. Specifically, we study the effects of unlabeled backdoor images under semi-supervised learning (SSL) on widely studied deep neural networks. To be realistic, we assume that the adversary is zero-knowledge and that the semi-supervised learning model is trained from scratch. Firstly, we find the fact that backdoor poisoning always fails when poisoned unlabeled images come from different classes, which is different from poisoning the labeled images. The reason is that the SSL algorithms always strive to correct them during training. Therefore, for unlabeled images, we implement backdoor poisoning on images from the target class. Then, we propose a gradient matching strategy to craft poisoned images such that their gradients match the gradients of target images on the SSL model, which can fit poisoned images to the target class and realize backdoor injection. To the best of our knowledge, this may be the first approach to backdoor poisoning on unlabeled images of trained-from-scratch SSL models. Experiments show that our poisoning achieves state-of-the-art attack success rates on most SSL algorithms while bypassing modern backdoor defenses.



## **25. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

cs.CV

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2210.08870v3) [paper-pdf](http://arxiv.org/pdf/2210.08870v3)

**Authors**: Jialiang Sun, Tingsong Jiang, Wen Yao, Donghua Wang, Xiaoqian Chen

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.



## **26. Generalizable Black-Box Adversarial Attack with Meta Learning**

cs.LG

T-PAMI 2022. Project Page is at https://github.com/SCLBD/MCG-Blackbox

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2301.00364v1) [paper-pdf](http://arxiv.org/pdf/2301.00364v1)

**Authors**: Fei Yin, Yong Zhang, Baoyuan Wu, Yan Feng, Jingyi Zhang, Yanbo Fan, Yujiu Yang

**Abstract**: In the scenario of black-box adversarial attack, the target model's parameters are unknown, and the attacker aims to find a successful adversarial perturbation based on query feedback under a query budget. Due to the limited feedback information, existing query-based black-box attack methods often require many queries for attacking each benign example. To reduce query cost, we propose to utilize the feedback information across historical attacks, dubbed example-level adversarial transferability. Specifically, by treating the attack on each benign example as one task, we develop a meta-learning framework by training a meta-generator to produce perturbations conditioned on benign examples. When attacking a new benign example, the meta generator can be quickly fine-tuned based on the feedback information of the new task as well as a few historical attacks to produce effective perturbations. Moreover, since the meta-train procedure consumes many queries to learn a generalizable generator, we utilize model-level adversarial transferability to train the meta-generator on a white-box surrogate model, then transfer it to help the attack against the target model. The proposed framework with the two types of adversarial transferability can be naturally combined with any off-the-shelf query-based attack methods to boost their performance, which is verified by extensive experiments.



## **27. ExploreADV: Towards exploratory attack for Neural Networks**

cs.CR

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2301.01223v1) [paper-pdf](http://arxiv.org/pdf/2301.01223v1)

**Authors**: Tianzuo Luo, Yuyi Zhong, Siaucheng Khoo

**Abstract**: Although deep learning has made remarkable progress in processing various types of data such as images, text and speech, they are known to be susceptible to adversarial perturbations: perturbations specifically designed and added to the input to make the target model produce erroneous output. Most of the existing studies on generating adversarial perturbations attempt to perturb the entire input indiscriminately. In this paper, we propose ExploreADV, a general and flexible adversarial attack system that is capable of modeling regional and imperceptible attacks, allowing users to explore various kinds of adversarial examples as needed. We adapt and combine two existing boundary attack methods, DeepFool and Brendel\&Bethge Attack, and propose a mask-constrained adversarial attack system, which generates minimal adversarial perturbations under the pixel-level constraints, namely ``mask-constraints''. We study different ways of generating such mask-constraints considering the variance and importance of the input features, and show that our adversarial attack system offers users good flexibility to focus on sub-regions of inputs, explore imperceptible perturbations and understand the vulnerability of pixels/regions to adversarial attacks. We demonstrate our system to be effective based on extensive experiments and user study.



## **28. WiFi Physical Layer Stays Awake and Responds When it Should Not**

cs.NI

12 pages

**SubmitDate**: 2022-12-31    [abs](http://arxiv.org/abs/2301.00269v1) [paper-pdf](http://arxiv.org/pdf/2301.00269v1)

**Authors**: Ali Abedi, Haofan Lu, Alex Chen, Charlie Liu, Omid Abari

**Abstract**: WiFi communication should be possible only between devices inside the same network. However, we find that all existing WiFi devices send back acknowledgments (ACK) to even fake packets received from unauthorized WiFi devices outside of their network. Moreover, we find that an unauthorized device can manipulate the power-saving mechanism of WiFi radios and keep them continuously awake by sending specific fake beacon frames to them. Our evaluation of over 5,000 devices from 186 vendors confirms that these are widespread issues. We believe these loopholes cannot be prevented, and hence they create privacy and security concerns. Finally, to show the importance of these issues and their consequences, we implement and demonstrate two attacks where an adversary performs battery drain and WiFi sensing attacks just using a tiny WiFi module which costs less than ten dollars.



## **29. A Comparative Study of Image Disguising Methods for Confidential Outsourced Learning**

cs.CR

**SubmitDate**: 2022-12-31    [abs](http://arxiv.org/abs/2301.00252v1) [paper-pdf](http://arxiv.org/pdf/2301.00252v1)

**Authors**: Sagar Sharma, Yuechun Gu, Keke Chen

**Abstract**: Large training data and expensive model tweaking are standard features of deep learning for images. As a result, data owners often utilize cloud resources to develop large-scale complex models, which raises privacy concerns. Existing solutions are either too expensive to be practical or do not sufficiently protect the confidentiality of data and models. In this paper, we study and compare novel \emph{image disguising} mechanisms, DisguisedNets and InstaHide, aiming to achieve a better trade-off among the level of protection for outsourced DNN model training, the expenses, and the utility of data. DisguisedNets are novel combinations of image blocktization, block-level random permutation, and two block-level secure transformations: random multidimensional projection (RMT) and AES pixel-level encryption (AES). InstaHide is an image mixup and random pixel flipping technique \cite{huang20}. We have analyzed and evaluated them under a multi-level threat model. RMT provides a better security guarantee than InstaHide, under the Level-1 adversarial knowledge with well-preserved model quality. In contrast, AES provides a security guarantee under the Level-2 adversarial knowledge, but it may affect model quality more. The unique features of image disguising also help us to protect models from model-targeted attacks. We have done an extensive experimental evaluation to understand how these methods work in different settings for different datasets.



## **30. FLAME: Taming Backdoors in Federated Learning**

cs.CR

To appear in the 31st USENIX Security Symposium, August 2022, Boston,  MA, USA

**SubmitDate**: 2022-12-31    [abs](http://arxiv.org/abs/2101.02281v4) [paper-pdf](http://arxiv.org/pdf/2101.02281v4)

**Authors**: Thien Duc Nguyen, Phillip Rieger, Huili Chen, Hossein Yalame, Helen Möllering, Hossein Fereidooni, Samuel Marchal, Markus Miettinen, Azalia Mirhoseini, Shaza Zeitouni, Farinaz Koushanfar, Ahmad-Reza Sadeghi, Thomas Schneider

**Abstract**: Federated Learning (FL) is a collaborative machine learning approach allowing participants to jointly train a model without having to share their private, potentially sensitive local datasets with others. Despite its benefits, FL is vulnerable to so-called backdoor attacks, in which an adversary injects manipulated model updates into the federated model aggregation process so that the resulting model will provide targeted false predictions for specific adversary-chosen inputs. Proposed defenses against backdoor attacks based on detecting and filtering out malicious model updates consider only very specific and limited attacker models, whereas defenses based on differential privacy-inspired noise injection significantly deteriorate the benign performance of the aggregated model. To address these deficiencies, we introduce FLAME, a defense framework that estimates the sufficient amount of noise to be injected to ensure the elimination of backdoors. To minimize the required amount of noise, FLAME uses a model clustering and weight clipping approach. This ensures that FLAME can maintain the benign performance of the aggregated model while effectively eliminating adversarial backdoors. Our evaluation of FLAME on several datasets stemming from application areas including image classification, word prediction, and IoT intrusion detection demonstrates that FLAME removes backdoors effectively with a negligible impact on the benign performance of the models.



## **31. Tracing the Origin of Adversarial Attack for Forensic Investigation and Deterrence**

cs.CR

**SubmitDate**: 2022-12-31    [abs](http://arxiv.org/abs/2301.01218v1) [paper-pdf](http://arxiv.org/pdf/2301.01218v1)

**Authors**: Han Fang, Jiyi Zhang, Yupeng Qiu, Ke Xu, Chengfang Fang, Ee-Chien Chang

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. In this paper, we take the role of investigators who want to trace the attack and identify the source, that is, the particular model which the adversarial examples are generated from. Techniques derived would aid forensic investigation of attack incidents and serve as deterrence to potential attacks. We consider the buyers-seller setting where a machine learning model is to be distributed to various buyers and each buyer receives a slightly different copy with same functionality. A malicious buyer generates adversarial examples from a particular copy $\mathcal{M}_i$ and uses them to attack other copies. From these adversarial examples, the investigator wants to identify the source $\mathcal{M}_i$. To address this problem, we propose a two-stage separate-and-trace framework. The model separation stage generates multiple copies of a model for a same classification task. This process injects unique characteristics into each copy so that adversarial examples generated have distinct and traceable features. We give a parallel structure which embeds a ``tracer'' in each copy, and a noise-sensitive training loss to achieve this goal. The tracing stage takes in adversarial examples and a few candidate models, and identifies the likely source. Based on the unique features induced by the noise-sensitive loss function, we could effectively trace the potential adversarial copy by considering the output logits from each tracer. Empirical results show that it is possible to trace the origin of the adversarial example and the mechanism can be applied to a wide range of architectures and datasets.



## **32. Guidance Through Surrogate: Towards a Generic Diagnostic Attack**

cs.LG

IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2212.14875v1) [paper-pdf](http://arxiv.org/pdf/2212.14875v1)

**Authors**: Muzammal Naseer, Salman Khan, Fatih Porikli, Fahad Shahbaz Khan

**Abstract**: Adversarial training is an effective approach to make deep neural networks robust against adversarial attacks. Recently, different adversarial training defenses are proposed that not only maintain a high clean accuracy but also show significant robustness against popular and well studied adversarial attacks such as PGD. High adversarial robustness can also arise if an attack fails to find adversarial gradient directions, a phenomenon known as `gradient masking'. In this work, we analyse the effect of label smoothing on adversarial training as one of the potential causes of gradient masking. We then develop a guided mechanism to avoid local minima during attack optimization, leading to a novel attack dubbed Guided Projected Gradient Attack (G-PGA). Our attack approach is based on a `match and deceive' loss that finds optimal adversarial directions through guidance from a surrogate model. Our modified attack does not require random restarts, large number of attack iterations or search for an optimal step-size. Furthermore, our proposed G-PGA is generic, thus it can be combined with an ensemble attack strategy as we demonstrate for the case of Auto-Attack, leading to efficiency and convergence speed improvements. More than an effective attack, G-PGA can be used as a diagnostic tool to reveal elusive robustness due to gradient masking in adversarial defenses.



## **33. Secure Fusion Estimation Against FDI Sensor Attacks in Cyber-Physical Systems**

eess.SY

10 pages, 5 figures; the first version of this manuscript was  completed on 2020

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2212.14755v1) [paper-pdf](http://arxiv.org/pdf/2212.14755v1)

**Authors**: Bo Chen, Pindi Weng, Daniel W. C. Ho, Li Yu

**Abstract**: This paper is concerned with the problem of secure multi-sensors fusion estimation for cyber-physical systems, where sensor measurements may be tampered with by false data injection (FDI) attacks. In this work, it is considered that the adversary may not be able to attack all sensors. That is, several sensors remain not being attacked. In this case, new local reorganized subsystems including the FDI attack signals and un-attacked sensor measurements are constructed by the augmentation method. Then, a joint Kalman fusion estimator is designed under linear minimum variance sense to estimate the system state and FDI attack signals simultaneously. Finally, illustrative examples are employed to show the effectiveness and advantages of the proposed methods.



## **34. Adversarial attacks and defenses on ML- and hardware-based IoT device fingerprinting and identification**

cs.CR

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2212.14677v1) [paper-pdf](http://arxiv.org/pdf/2212.14677v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Gérôme Bovet, Gregorio Martínez Pérez

**Abstract**: In the last years, the number of IoT devices deployed has suffered an undoubted explosion, reaching the scale of billions. However, some new cybersecurity issues have appeared together with this development. Some of these issues are the deployment of unauthorized devices, malicious code modification, malware deployment, or vulnerability exploitation. This fact has motivated the requirement for new device identification mechanisms based on behavior monitoring. Besides, these solutions have recently leveraged Machine and Deep Learning techniques due to the advances in this field and the increase in processing capabilities. In contrast, attackers do not stay stalled and have developed adversarial attacks focused on context modification and ML/DL evaluation evasion applied to IoT device identification solutions. This work explores the performance of hardware behavior-based individual device identification, how it is affected by possible context- and ML/DL-focused attacks, and how its resilience can be improved using defense techniques. In this sense, it proposes an LSTM-CNN architecture based on hardware performance behavior for individual device identification. Then, previous techniques have been compared with the proposed architecture using a hardware performance dataset collected from 45 Raspberry Pi devices running identical software. The LSTM-CNN improves previous solutions achieving a +0.96 average F1-Score and 0.8 minimum TPR for all devices. Afterward, context- and ML/DL-focused adversarial attacks were applied against the previous model to test its robustness. A temperature-based context attack was not able to disrupt the identification. However, some ML/DL state-of-the-art evasion attacks were successful. Finally, adversarial training and model distillation defense techniques are selected to improve the model resilience to evasion attacks, without degrading its performance.



## **35. Defense Against Adversarial Attacks on Audio DeepFake Detection**

cs.SD

Submitted to ICASSP 2023

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2212.14597v1) [paper-pdf](http://arxiv.org/pdf/2212.14597v1)

**Authors**: Piotr Kawa, Marcin Plata, Piotr Syga

**Abstract**: Audio DeepFakes are artificially generated utterances created using deep learning methods with the main aim to fool the listeners, most of such audio is highly convincing. Their quality is sufficient to pose a serious threat in terms of security and privacy, such as the reliability of news or defamation. To prevent the threats, multiple neural networks-based methods to detect generated speech have been proposed. In this work, we cover the topic of adversarial attacks, which decrease the performance of detectors by adding superficial (difficult to spot by a human) changes to input data. Our contribution contains evaluating the robustness of 3 detection architectures against adversarial attacks in two scenarios (white-box and using transferability mechanism) and enhancing it later by the use of adversarial training performed by our novel adaptive training method.



## **36. CARE: Certifiably Robust Learning with Reasoning via Variational Inference**

cs.LG

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2209.05055v3) [paper-pdf](http://arxiv.org/pdf/2209.05055v3)

**Authors**: Jiawei Zhang, Linyi Li, Ce Zhang, Bo Li

**Abstract**: Despite great recent advances achieved by deep neural networks (DNNs), they are often vulnerable to adversarial attacks. Intensive research efforts have been made to improve the robustness of DNNs; however, most empirical defenses can be adaptively attacked again, and the theoretically certified robustness is limited, especially on large-scale datasets. One potential root cause of such vulnerabilities for DNNs is that although they have demonstrated powerful expressiveness, they lack the reasoning ability to make robust and reliable predictions. In this paper, we aim to integrate domain knowledge to enable robust learning with the reasoning paradigm. In particular, we propose a certifiably robust learning with reasoning pipeline (CARE), which consists of a learning component and a reasoning component. Concretely, we use a set of standard DNNs to serve as the learning component to make semantic predictions, and we leverage the probabilistic graphical models, such as Markov logic networks (MLN), to serve as the reasoning component to enable knowledge/logic reasoning. However, it is known that the exact inference of MLN (reasoning) is #P-complete, which limits the scalability of the pipeline. To this end, we propose to approximate the MLN inference via variational inference based on an efficient expectation maximization algorithm. In particular, we leverage graph convolutional networks (GCNs) to encode the posterior distribution during variational inference and update the parameters of GCNs (E-step) and the weights of knowledge rules in MLN (M-step) iteratively. We conduct extensive experiments on different datasets and show that CARE achieves significantly higher certified robustness compared with the state-of-the-art baselines. We additionally conducted different ablation studies to demonstrate the empirical robustness of CARE and the effectiveness of different knowledge integration.



## **37. "Real Attackers Don't Compute Gradients": Bridging the Gap Between Adversarial ML Research and Practice**

cs.CR

**SubmitDate**: 2022-12-29    [abs](http://arxiv.org/abs/2212.14315v1) [paper-pdf](http://arxiv.org/pdf/2212.14315v1)

**Authors**: Giovanni Apruzzese, Hyrum S. Anderson, Savino Dambra, David Freeman, Fabio Pierazzi, Kevin A. Roundy

**Abstract**: Recent years have seen a proliferation of research on adversarial machine learning. Numerous papers demonstrate powerful algorithmic attacks against a wide variety of machine learning (ML) models, and numerous other papers propose defenses that can withstand most attacks. However, abundant real-world evidence suggests that actual attackers use simple tactics to subvert ML-driven systems, and as a result security practitioners have not prioritized adversarial ML defenses.   Motivated by the apparent gap between researchers and practitioners, this position paper aims to bridge the two domains. We first present three real-world case studies from which we can glean practical insights unknown or neglected in research. Next we analyze all adversarial ML papers recently published in top security conferences, highlighting positive trends and blind spots. Finally, we state positions on precise and cost-driven threat modeling, collaboration between industry and academia, and reproducible research. We believe that our positions, if adopted, will increase the real-world impact of future endeavours in adversarial ML, bringing both researchers and practitioners closer to their shared goal of improving the security of ML systems.



## **38. Towards Comprehensively Understanding the Run-time Security of Programmable Logic Controllers: A 3-year Empirical Study**

cs.CR

**SubmitDate**: 2022-12-29    [abs](http://arxiv.org/abs/2212.14296v1) [paper-pdf](http://arxiv.org/pdf/2212.14296v1)

**Authors**: Rongkuan Ma, Qiang Wei, Jingyi Wang, Shunkai Zhu, Shouling Ji, Peng Cheng, Yan Jia, Qingxian Wang

**Abstract**: Programmable Logic Controllers (PLCs) are the core control devices in Industrial Control Systems (ICSs), which control and monitor the underlying physical plants such as power grids. PLCs were initially designed to work in a trusted industrial network, which however can be brittle once deployed in an Internet-facing (or penetrated) network. Yet, there is a lack of systematic empirical analysis of the run-time security of modern real-world PLCs. To close this gap, we present the first large-scale measurement on 23 off-the-shelf PLCs across 13 leading vendors. We find many common security issues and unexplored implications that should be more carefully addressed in the design and implementation. To sum up, the unsupervised logic applications can cause system resource/privilege abuse, which gives adversaries new means to hijack the control flow of a runtime system remotely (without exploiting memory vulnerabilities); 2) the improper access control mechanisms bring many unauthorized access implications; 3) the proprietary or semi-proprietary protocols are fragile regarding confidentiality and integrity protection of run-time data. We empirically evaluated the corresponding attack vectors on multiple PLCs, which demonstrates that the security implications are severe and broad. Our findings were reported to the related parties responsibly, and 20 bugs have been confirmed with 7 assigned CVEs.



## **39. Attention, Please! Adversarial Defense via Activation Rectification and Preservation**

cs.CV

**SubmitDate**: 2022-12-29    [abs](http://arxiv.org/abs/1811.09831v5) [paper-pdf](http://arxiv.org/pdf/1811.09831v5)

**Authors**: Shangxi Wu, Jitao Sang, Kaiyuan Xu, Jiaming Zhang, Jian Yu

**Abstract**: This study provides a new understanding of the adversarial attack problem by examining the correlation between adversarial attack and visual attention change. In particular, we observed that: (1) images with incomplete attention regions are more vulnerable to adversarial attacks; and (2) successful adversarial attacks lead to deviated and scattered attention map. Accordingly, an attention-based adversarial defense framework is designed to simultaneously rectify the attention map for prediction and preserve the attention area between adversarial and original images. The problem of adding iteratively attacked samples is also discussed in the context of visual attention change. We hope the attention-related data analysis and defense solution in this study will shed some light on the mechanism behind the adversarial attack and also facilitate future adversarial defense/attack model design.



## **40. Reducing Certified Regression to Certified Classification for General Poisoning Attacks**

cs.LG

Accepted at the 1st IEEE conference on Secure and Trustworthy Machine  Learning (SaTML'23)

**SubmitDate**: 2022-12-29    [abs](http://arxiv.org/abs/2208.13904v2) [paper-pdf](http://arxiv.org/pdf/2208.13904v2)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Adversarial training instances can severely distort a model's behavior. This work investigates certified regression defenses, which provide guaranteed limits on how much a regressor's prediction may change under a poisoning attack. Our key insight is that certified regression reduces to voting-based certified classification when using median as a model's primary decision function. Coupling our reduction with existing certified classifiers, we propose six new regressors provably-robust to poisoning attacks. To the extent of our knowledge, this is the first work that certifies the robustness of individual regression predictions without any assumptions about the data distribution and model architecture. We also show that the assumptions made by existing state-of-the-art certified classifiers are often overly pessimistic. We introduce a tighter analysis of model robustness, which in many cases results in significantly improved certified guarantees. Lastly, we empirically demonstrate our approaches' effectiveness on both regression and classification data, where the accuracy of up to 50% of test predictions can be guaranteed under 1% training set corruption and up to 30% of predictions under 4% corruption. Our source code is available at https://github.com/ZaydH/certified-regression.



## **41. Certifying Safety in Reinforcement Learning under Adversarial Perturbation Attacks**

cs.LG

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.14115v1) [paper-pdf](http://arxiv.org/pdf/2212.14115v1)

**Authors**: Junlin Wu, Hussein Sibai, Yevgeniy Vorobeychik

**Abstract**: Function approximation has enabled remarkable advances in applying reinforcement learning (RL) techniques in environments with high-dimensional inputs, such as images, in an end-to-end fashion, mapping such inputs directly to low-level control. Nevertheless, these have proved vulnerable to small adversarial input perturbations. A number of approaches for improving or certifying robustness of end-to-end RL to adversarial perturbations have emerged as a result, focusing on cumulative reward. However, what is often at stake in adversarial scenarios is the violation of fundamental properties, such as safety, rather than the overall reward that combines safety with efficiency. Moreover, properties such as safety can only be defined with respect to true state, rather than the high-dimensional raw inputs to end-to-end policies. To disentangle nominal efficiency and adversarial safety, we situate RL in deterministic partially-observable Markov decision processes (POMDPs) with the goal of maximizing cumulative reward subject to safety constraints. We then propose a partially-supervised reinforcement learning (PSRL) framework that takes advantage of an additional assumption that the true state of the POMDP is known at training time. We present the first approach for certifying safety of PSRL policies under adversarial input perturbations, and two adversarial training approaches that make direct use of PSRL. Our experiments demonstrate both the efficacy of the proposed approach for certifying safety in adversarial environments, and the value of the PSRL framework coupled with adversarial training in improving certified safety while preserving high nominal reward and high-quality predictions of true state.



## **42. Robust Ranking Explanations**

cs.LG

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.14106v1) [paper-pdf](http://arxiv.org/pdf/2212.14106v1)

**Authors**: Chao Chen, Chenghua Guo, Guixiang Ma, Xi Zhang, Sihong Xie

**Abstract**: Gradient-based explanation is the cornerstone of explainable deep networks, but it has been shown to be vulnerable to adversarial attacks. However, existing works measure the explanation robustness based on $\ell_p$-norm, which can be counter-intuitive to humans, who only pay attention to the top few salient features. We propose explanation ranking thickness as a more suitable explanation robustness metric. We then present a new practical adversarial attacking goal for manipulating explanation rankings. To mitigate the ranking-based attacks while maintaining computational feasibility, we derive surrogate bounds of the thickness that involve expensive sampling and integration. We use a multi-objective approach to analyze the convergence of a gradient-based attack to confirm that the explanation robustness can be measured by the thickness metric. We conduct experiments on various network architectures and diverse datasets to prove the superiority of the proposed methods, while the widely accepted Hessian-based curvature smoothing approaches are not as robust as our method.



## **43. ObjectSeeker: Certifiably Robust Object Detection against Patch Hiding Attacks via Patch-agnostic Masking**

cs.CV

IEEE Symposium on Security and Privacy 2023; extended version

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2202.01811v2) [paper-pdf](http://arxiv.org/pdf/2202.01811v2)

**Authors**: Chong Xiang, Alexander Valtchanov, Saeed Mahloujifar, Prateek Mittal

**Abstract**: Object detectors, which are widely deployed in security-critical systems such as autonomous vehicles, have been found vulnerable to patch hiding attacks. An attacker can use a single physically-realizable adversarial patch to make the object detector miss the detection of victim objects and undermine the functionality of object detection applications. In this paper, we propose ObjectSeeker for certifiably robust object detection against patch hiding attacks. The key insight in ObjectSeeker is patch-agnostic masking: we aim to mask out the entire adversarial patch without knowing the shape, size, and location of the patch. This masking operation neutralizes the adversarial effect and allows any vanilla object detector to safely detect objects on the masked images. Remarkably, we can evaluate ObjectSeeker's robustness in a certifiable manner: we develop a certification procedure to formally determine if ObjectSeeker can detect certain objects against any white-box adaptive attack within the threat model, achieving certifiable robustness. Our experiments demonstrate a significant (~10%-40% absolute and ~2-6x relative) improvement in certifiable robustness over the prior work, as well as high clean performance (~1% drop compared with undefended models).



## **44. Internal Wasserstein Distance for Adversarial Attack and Defense**

cs.LG

Due to the need for major revisions to the submitted manuscript, we  request to withdraw this manuscript from arXiv

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2103.07598v3) [paper-pdf](http://arxiv.org/pdf/2103.07598v3)

**Authors**: Mingkui Tan, Shuhai Zhang, Jiezhang Cao, Jincheng Li, Yanwu Xu

**Abstract**: Deep neural networks (DNNs) are known to be vulnerable to adversarial attacks that would trigger misclassification of DNNs but may be imperceptible to human perception. Adversarial defense has been important ways to improve the robustness of DNNs. Existing attack methods often construct adversarial examples relying on some metrics like the $\ell_p$ distance to perturb samples. However, these metrics can be insufficient to conduct adversarial attacks due to their limited perturbations. In this paper, we propose a new internal Wasserstein distance (IWD) to capture the semantic similarity of two samples, and thus it helps to obtain larger perturbations than currently used metrics such as the $\ell_p$ distance We then apply the internal Wasserstein distance to perform adversarial attack and defense. In particular, we develop a novel attack method relying on IWD to calculate the similarities between an image and its adversarial examples. In this way, we can generate diverse and semantically similar adversarial examples that are more difficult to defend by existing defense methods. Moreover, we devise a new defense method relying on IWD to learn robust models against unseen adversarial examples. We provide both thorough theoretical and empirical evidence to support our methods.



## **45. Thermal Heating in ReRAM Crossbar Arrays: Challenges and Solutions**

cs.AR

18 pages

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.13707v1) [paper-pdf](http://arxiv.org/pdf/2212.13707v1)

**Authors**: Kamilya Smagulova, Mohammed E. Fouda, Ahmed Eltawil

**Abstract**: Increasing popularity of deep-learning-powered applications raises the issue of vulnerability of neural networks to adversarial attacks. In other words, hardly perceptible changes in input data lead to the output error in neural network hindering their utilization in applications that involve decisions with security risks. A number of previous works have already thoroughly evaluated the most commonly used configuration - Convolutional Neural Networks (CNNs) against different types of adversarial attacks. Moreover, recent works demonstrated transferability of the some adversarial examples across different neural network models. This paper studied robustness of the new emerging models such as SpinalNet-based neural networks and Compact Convolutional Transformers (CCT) on image classification problem of CIFAR-10 dataset. Each architecture was tested against four White-box attacks and three Black-box attacks. Unlike VGG and SpinalNet models, attention-based CCT configuration demonstrated large span between strong robustness and vulnerability to adversarial examples. Eventually, the study of transferability between VGG, VGG-inspired SpinalNet and pretrained CCT 7/3x1 models was conducted. It was shown that despite high effectiveness of the attack on the certain individual model, this does not guarantee the transferability to other models.



## **46. Publishing Efficient On-device Models Increases Adversarial Vulnerability**

cs.CR

Accepted to IEEE SaTML 2023

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.13700v1) [paper-pdf](http://arxiv.org/pdf/2212.13700v1)

**Authors**: Sanghyun Hong, Nicholas Carlini, Alexey Kurakin

**Abstract**: Recent increases in the computational demands of deep neural networks (DNNs) have sparked interest in efficient deep learning mechanisms, e.g., quantization or pruning. These mechanisms enable the construction of a small, efficient version of commercial-scale models with comparable accuracy, accelerating their deployment to resource-constrained devices.   In this paper, we study the security considerations of publishing on-device variants of large-scale models. We first show that an adversary can exploit on-device models to make attacking the large models easier. In evaluations across 19 DNNs, by exploiting the published on-device models as a transfer prior, the adversarial vulnerability of the original commercial-scale models increases by up to 100x. We then show that the vulnerability increases as the similarity between a full-scale and its efficient model increase. Based on the insights, we propose a defense, $similarity$-$unpairing$, that fine-tunes on-device models with the objective of reducing the similarity. We evaluated our defense on all the 19 DNNs and found that it reduces the transferability up to 90% and the number of queries required by a factor of 10-100x. Our results suggest that further research is needed on the security (or even privacy) threats caused by publishing those efficient siblings.



## **47. Learning When to Use Adaptive Adversarial Image Perturbations against Autonomous Vehicles**

cs.RO

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.13667v1) [paper-pdf](http://arxiv.org/pdf/2212.13667v1)

**Authors**: Hyung-Jin Yoon, Hamidreza Jafarnejadsani, Petros Voulgaris

**Abstract**: The deep neural network (DNN) models for object detection using camera images are widely adopted in autonomous vehicles. However, DNN models are shown to be susceptible to adversarial image perturbations. In the existing methods of generating the adversarial image perturbations, optimizations take each incoming image frame as the decision variable to generate an image perturbation. Therefore, given a new image, the typically computationally-expensive optimization needs to start over as there is no learning between the independent optimizations. Very few approaches have been developed for attacking online image streams while considering the underlying physical dynamics of autonomous vehicles, their mission, and the environment. We propose a multi-level stochastic optimization framework that monitors an attacker's capability of generating the adversarial perturbations. Based on this capability level, a binary decision attack/not attack is introduced to enhance the effectiveness of the attacker. We evaluate our proposed multi-level image attack framework using simulations for vision-guided autonomous vehicles and actual tests with a small indoor drone in an office environment. The results show our method's capability to generate the image attack in real-time while monitoring when the attacker is proficient given state estimates.



## **48. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

cs.CR

12 pages, 7 figures, 5 tables

**SubmitDate**: 2022-12-27    [abs](http://arxiv.org/abs/2204.11307v3) [paper-pdf](http://arxiv.org/pdf/2204.11307v3)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstract**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.



## **49. EDoG: Adversarial Edge Detection For Graph Neural Networks**

cs.LG

Accepted by IEEE Conference on Secure and Trustworthy Machine  Learning 2023

**SubmitDate**: 2022-12-27    [abs](http://arxiv.org/abs/2212.13607v1) [paper-pdf](http://arxiv.org/pdf/2212.13607v1)

**Authors**: Xiaojun Xu, Yue Yu, Hanzhang Wang, Alok Lal, Carl A. Gunter, Bo Li

**Abstract**: Graph Neural Networks (GNNs) have been widely applied to different tasks such as bioinformatics, drug design, and social networks. However, recent studies have shown that GNNs are vulnerable to adversarial attacks which aim to mislead the node or subgraph classification prediction by adding subtle perturbations. Detecting these attacks is challenging due to the small magnitude of perturbation and the discrete nature of graph data. In this paper, we propose a general adversarial edge detection pipeline EDoG without requiring knowledge of the attack strategies based on graph generation. Specifically, we propose a novel graph generation approach combined with link prediction to detect suspicious adversarial edges. To effectively train the graph generative model, we sample several sub-graphs from the given graph data. We show that since the number of adversarial edges is usually low in practice, with low probability the sampled sub-graphs will contain adversarial edges based on the union bound. In addition, considering the strong attacks which perturb a large number of edges, we propose a set of novel features to perform outlier detection as the preprocessing for our detection. Extensive experimental results on three real-world graph datasets including a private transaction rule dataset from a major company and two types of synthetic graphs with controlled properties show that EDoG can achieve above 0.8 AUC against four state-of-the-art unseen attack strategies without requiring any knowledge about the attack type; and around 0.85 with knowledge of the attack type. EDoG significantly outperforms traditional malicious edge detection baselines. We also show that an adaptive attack with full knowledge of our detection pipeline is difficult to bypass it.



## **50. Towards Transferable Unrestricted Adversarial Examples with Minimum Changes**

cs.CV

Accepted at SaTML 2023

**SubmitDate**: 2022-12-27    [abs](http://arxiv.org/abs/2201.01102v2) [paper-pdf](http://arxiv.org/pdf/2201.01102v2)

**Authors**: Fangcheng Liu, Chao Zhang, Hongyang Zhang

**Abstract**: Transfer-based adversarial example is one of the most important classes of black-box attacks. However, there is a trade-off between transferability and imperceptibility of the adversarial perturbation. Prior work in this direction often requires a fixed but large $\ell_p$-norm perturbation budget to reach a good transfer success rate, leading to perceptible adversarial perturbations. On the other hand, most of the current unrestricted adversarial attacks that aim to generate semantic-preserving perturbations suffer from weaker transferability to the target model. In this work, we propose a geometry-aware framework to generate transferable adversarial examples with minimum changes. Analogous to model selection in statistical machine learning, we leverage a validation model to select the best perturbation budget for each image under both the $\ell_{\infty}$-norm and unrestricted threat models. We propose a principled method for the partition of training and validation models by encouraging intra-group diversity while penalizing extra-group similarity. Extensive experiments verify the effectiveness of our framework on balancing imperceptibility and transferability of the crafted adversarial examples. The methodology is the foundation of our entry to the CVPR'21 Security AI Challenger: Unrestricted Adversarial Attacks on ImageNet, in which we ranked 1st place out of 1,559 teams and surpassed the runner-up submissions by 4.59% and 23.91% in terms of final score and average image quality level, respectively. Code is available at https://github.com/Equationliu/GA-Attack.



