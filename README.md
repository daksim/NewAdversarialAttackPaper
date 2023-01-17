# Latest Adversarial Attack Papers
**update at 2023-01-17 14:00:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Threat Models over Space and Time: A Case Study of E2EE Messaging Applications**

cs.CR

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05653v1) [paper-pdf](http://arxiv.org/pdf/2301.05653v1)

**Authors**: Partha Das Chowdhury, Maria Sameen, Jenny Blessing, Nicholas Boucher, Joseph Gardiner, Tom Burrows, Ross Anderson, Awais Rashid

**Abstract**: Threat modelling is foundational to secure systems engineering and should be done in consideration of the context within which systems operate. On the other hand, the continuous evolution of both the technical sophistication of threats and the system attack surface is an inescapable reality. In this work, we explore the extent to which real-world systems engineering reflects the changing threat context. To this end we examine the desktop clients of six widely used end-to-end-encrypted mobile messaging applications to understand the extent to which they adjusted their threat model over space (when enabling clients on new platforms, such as desktop clients) and time (as new threats emerged). We experimented with short-lived adversarial access against these desktop clients and analyzed the results with respect to two popular threat elicitation frameworks, STRIDE and LINDDUN. The results demonstrate that system designers need to both recognise the threats in the evolving context within which systems operate and, more importantly, to mitigate them by rescoping trust boundaries in a manner that those within the administrative boundary cannot violate security and privacy properties. Such a nuanced understanding of trust boundary scopes and their relationship with administrative boundaries allows for better administration of shared components, including securing them with safe defaults.



## **2. Resilient Model Predictive Control of Distributed Systems Under Attack Using Local Attack Identification**

cs.SY

Submitted for review to Springer Natural Computer Science on November  18th 2022

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05547v1) [paper-pdf](http://arxiv.org/pdf/2301.05547v1)

**Authors**: Sarah Braun, Sebastian Albrecht, Sergio Lucia

**Abstract**: With the growing share of renewable energy sources, the uncertainty in power supply is increasing. In addition to the inherent fluctuations in the renewables, this is due to the threat of deliberate malicious attacks, which may become more revalent with a growing number of distributed generation units. Also in other safety-critical technology sectors, control systems are becoming more and more decentralized, causing the targets for attackers and thus the risk of attacks to increase. It is thus essential that distributed controllers are robust toward these uncertainties and able to react quickly to disturbances of any kind. To this end, we present novel methods for model-based identification of attacks and combine them with distributed model predictive control to obtain a resilient framework for adaptively robust control. The methodology is specially designed for distributed setups with limited local information due to privacy and security reasons. To demonstrate the efficiency of the method, we introduce a mathematical model for physically coupled microgrids under the uncertain influence of renewable generation and adversarial attacks, and perform numerical experiments, applying the proposed method for microgrid control.



## **3. PMFault: Faulting and Bricking Server CPUs through Management Interfaces**

cs.CR

For demo and source code, visit https://zt-chen.github.io/PMFault/

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05538v1) [paper-pdf](http://arxiv.org/pdf/2301.05538v1)

**Authors**: Zitai Chen, David Oswald

**Abstract**: Apart from the actual CPU, modern server motherboards contain other auxiliary components, for example voltage regulators for power management. Those are connected to the CPU and the separate Baseboard Management Controller (BMC) via the I2C-based PMBus.   In this paper, using the case study of the widely used Supermicro X11SSL motherboard, we show how remotely exploitable software weaknesses in the BMC (or other processors with PMBus access) can be used to access the PMBus and then perform hardware-based fault injection attacks on the main CPU. The underlying weaknesses include insecure firmware encryption and signing mechanisms, a lack of authentication for the firmware upgrade process and the IPMI KCS control interface, as well as the motherboard design (with the PMBus connected to the BMC and SMBus by default).   First, we show that undervolting through the PMBus allows breaking the integrity guarantees of SGX enclaves, bypassing Intel's countermeasures against previous undervolting attacks like Plundervolt/V0ltPwn. Second, we experimentally show that overvolting outside the specified range has the potential of permanently damaging Intel Xeon CPUs, rendering the server inoperable. We assess the impact of our findings on other server motherboards made by Supermicro and ASRock.   Our attacks, dubbed PMFault, can be carried out by a privileged software adversary and do not require physical access to the server motherboard or knowledge of the BMC login credentials.   We responsibly disclosed the issues reported in this paper to Supermicro and discuss possible countermeasures at different levels. To the best of our knowledge, the 12th generation of Supermicro motherboards, which was designed before we reported PMFault to Supermicro, is not vulnerable.



## **4. On the feasibility of attacking Thai LPR systems with adversarial examples**

cs.CR

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05506v1) [paper-pdf](http://arxiv.org/pdf/2301.05506v1)

**Authors**: Chissanupong Jiamsuchon, Jakapan Suaboot, Norrathep Rattanavipanon

**Abstract**: Recent advances in deep neural networks (DNNs) have significantly enhanced the capabilities of optical character recognition (OCR) technology, enabling its adoption to a wide range of real-world applications. Despite this success, DNN-based OCR is shown to be vulnerable to adversarial attacks, in which the adversary can influence the DNN model's prediction by carefully manipulating input to the model. Prior work has demonstrated the security impacts of adversarial attacks on various OCR languages. However, to date, no studies have been conducted and evaluated on an OCR system tailored specifically for the Thai language. To bridge this gap, this work presents a feasibility study of performing adversarial attacks on a specific Thai OCR application -- Thai License Plate Recognition (LPR). Moreover, we propose a new type of adversarial attack based on the \emph{semi-targeted} scenario and show that this scenario is highly realistic in LPR applications. Our experimental results show the feasibility of our attacks as they can be performed on a commodity computer desktop with over 90% attack success rate.



## **5. Feature Importance Guided Attack: A Model Agnostic Adversarial Attack**

cs.LG

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2106.14815v3) [paper-pdf](http://arxiv.org/pdf/2106.14815v3)

**Authors**: Gilad Gressel, Niranjan Hegde, Archana Sreekumar, Rishikumar Radhakrishnan, Kalyani Harikumar, Anjali S., Krishnashree Achuthan

**Abstract**: Research in adversarial learning has primarily focused on homogeneous unstructured datasets, which often map into the problem space naturally. Inverting a feature space attack on heterogeneous datasets into the problem space is much more challenging, particularly the task of finding the perturbation to perform. This work presents a formal search strategy: the `Feature Importance Guided Attack' (FIGA), which finds perturbations in the feature space of heterogeneous tabular datasets to produce evasion attacks. We first demonstrate FIGA in the feature space and then in the problem space. FIGA assumes no prior knowledge of the defending model's learning algorithm and does not require any gradient information. FIGA assumes knowledge of the feature representation and the mean feature values of defending model's dataset. FIGA leverages feature importance rankings by perturbing the most important features of the input in the direction of the target class. While FIGA is conceptually similar to other work which uses feature selection processes (e.g., mimicry attacks), we formalize an attack algorithm with three tunable parameters and investigate the strength of FIGA on tabular datasets. We demonstrate the effectiveness of FIGA by evading phishing detection models trained on four different tabular phishing datasets and one financial dataset with an average success rate of 94%. We extend FIGA to the phishing problem space by limiting the possible perturbations to be valid and feasible in the phishing domain. We generate valid adversarial phishing sites that are visually identical to their unperturbed counterpart and use them to attack six tabular ML models achieving a 13.05% average success rate.



## **6. Security-Aware Approximate Spiking Neural Networks**

cs.CR

Accepted full paper in DATE 2023

**SubmitDate**: 2023-01-12    [abs](http://arxiv.org/abs/2301.05264v1) [paper-pdf](http://arxiv.org/pdf/2301.05264v1)

**Authors**: Syed Tihaam Ahmad, Ayesha Siddique, Khaza Anuarul Hoque

**Abstract**: Deep Neural Networks (DNNs) and Spiking Neural Networks (SNNs) are both known for their susceptibility to adversarial attacks. Therefore, researchers in the recent past have extensively studied the robustness and defense of DNNs and SNNs under adversarial attacks. Compared to accurate SNNs (AccSNN), approximate SNNs (AxSNNs) are known to be up to 4X more energy-efficient for ultra-low power applications. Unfortunately, the robustness of AxSNNs under adversarial attacks is yet unexplored. In this paper, we first extensively analyze the robustness of AxSNNs with different structural parameters and approximation levels under two gradient-based and two neuromorphic attacks. Then, we propose two novel defense methods, i.e., precision scaling and approximate quantization-aware filtering (AQF), for securing AxSNNs. We evaluated the effectiveness of these two defense methods using both static and neuromorphic datasets. Our results demonstrate that AxSNNs are more prone to adversarial attacks than AccSNNs, but precision scaling and AQF significantly improve the robustness of AxSNNs. For instance, a PGD attack on AxSNN results in a 72\% accuracy loss compared to AccSNN without any attack, whereas the same attack on the precision-scaled AxSNN leads to only a 17\% accuracy loss in the static MNIST dataset (4X robustness improvement). Similarly, a Sparse Attack on AxSNN leads to a 77\% accuracy loss when compared to AccSNN without any attack, whereas the same attack on an AxSNN with AQF leads to only a 2\% accuracy loss in the neuromorphic DVS128 Gesture dataset (38X robustness improvement).



## **7. Jamming Attacks on Decentralized Federated Learning in General Multi-Hop Wireless Networks**

cs.NI

**SubmitDate**: 2023-01-12    [abs](http://arxiv.org/abs/2301.05250v1) [paper-pdf](http://arxiv.org/pdf/2301.05250v1)

**Authors**: Yi Shi, Yalin E. Sagduyu, Tugba Erpek

**Abstract**: Decentralized federated learning (DFL) is an effective approach to train a deep learning model at multiple nodes over a multi-hop network, without the need of a server having direct connections to all nodes. In general, as long as nodes are connected potentially via multiple hops, the DFL process will eventually allow each node to experience the effects of models from all other nodes via either direct connections or multi-hop paths, and thus is able to train a high-fidelity model at each node. We consider an effective attack that uses jammers to prevent the model exchanges between nodes. There are two attack scenarios. First, the adversary can attack any link under a certain budget. Once attacked, two end nodes of a link cannot exchange their models. Secondly, some jammers with limited jamming ranges are deployed in the network and a jammer can only jam nodes within its jamming range. Once a directional link is attacked, the receiver node cannot receive the model from the transmitter node. We design algorithms to select links to be attacked for both scenarios. For the second scenario, we also design algorithms to deploy jammers at optimal locations so that they can attack critical nodes and achieve the highest impact on the DFL process. We evaluate these algorithms by using wireless signal classification over a large network area as the use case and identify how these attack mechanisms exploits various learning, connectivity, and sensing aspects. We show that the DFL performance can be significantly reduced by jamming attacks launched in a wireless network and characterize the attack surface as a vulnerability study before the safe deployment of DFL over wireless networks.



## **8. Phase-shifted Adversarial Training**

cs.LG

**SubmitDate**: 2023-01-12    [abs](http://arxiv.org/abs/2301.04785v1) [paper-pdf](http://arxiv.org/pdf/2301.04785v1)

**Authors**: Yeachan Kim, Seongyeon Kim, Ihyeok Seo, Bonggun Shin

**Abstract**: Adversarial training has been considered an imperative component for safely deploying neural network-based applications to the real world. To achieve stronger robustness, existing methods primarily focus on how to generate strong attacks by increasing the number of update steps, regularizing the models with the smoothed loss function, and injecting the randomness into the attack. Instead, we analyze the behavior of adversarial training through the lens of response frequency. We empirically discover that adversarial training causes neural networks to have low convergence to high-frequency information, resulting in highly oscillated predictions near each data. To learn high-frequency contents efficiently and effectively, we first prove that a universal phenomenon of frequency principle, i.e., \textit{lower frequencies are learned first}, still holds in adversarial training. Based on that, we propose phase-shifted adversarial training (PhaseAT) in which the model learns high-frequency components by shifting these frequencies to the low-frequency range where the fast convergence occurs. For evaluations, we conduct the experiments on CIFAR-10 and ImageNet with the adaptive attack carefully designed for reliable evaluation. Comprehensive results show that PhaseAT significantly improves the convergence for high-frequency information. This results in improved adversarial robustness by enabling the model to have smoothed predictions near each data.



## **9. Reducing Exploitability with Population Based Training**

cs.LG

Presented at New Frontiers in Adversarial Machine Learning Workshop,  ICML 2022

**SubmitDate**: 2023-01-11    [abs](http://arxiv.org/abs/2208.05083v3) [paper-pdf](http://arxiv.org/pdf/2208.05083v3)

**Authors**: Pavel Czempin, Adam Gleave

**Abstract**: Self-play reinforcement learning has achieved state-of-the-art, and often superhuman, performance in a variety of zero-sum games. Yet prior work has found that policies that are highly capable against regular opponents can fail catastrophically against adversarial policies: an opponent trained explicitly against the victim. Prior defenses using adversarial training were able to make the victim robust to a specific adversary, but the victim remained vulnerable to new ones. We conjecture this limitation was due to insufficient diversity of adversaries seen during training. We analyze a defense using population based training to pit the victim against a diverse set of opponents. We evaluate this defense's robustness against new adversaries in two low-dimensional environments. This defense increases robustness against adversaries, as measured by the number of attacker training timesteps to exploit the victim. Furthermore, we show that robustness is correlated with the size of the opponent population.



## **10. Resynthesis-based Attacks Against Logic Locking**

cs.CR

8 pages, 7 figures, conference

**SubmitDate**: 2023-01-11    [abs](http://arxiv.org/abs/2301.04400v1) [paper-pdf](http://arxiv.org/pdf/2301.04400v1)

**Authors**: F. Almeida, L. Aksoy, Q-L. Nguyen, S. Dupuis, M-L. Flottes, S. Pagliarini

**Abstract**: Logic locking has been a promising solution to many hardware security threats, such as intellectual property infringement and overproduction. Due to the increased attention that threats have received, many efficient specialized attacks against logic locking have been introduced over the years. However, the ability of an adversary to manipulate a locked netlist prior to mounting an attack has not been investigated thoroughly. This paper introduces a resynthesis-based strategy that utilizes the strength of a commercial electronic design automation (EDA) tool to reveal the vulnerabilities of a locked circuit. To do so, in a pre-attack step, a locked netlist is resynthesized using different synthesis parameters in a systematic way, leading to a large number of functionally equivalent but structurally different locked circuits. Then, under the oracle-less threat model, where it is assumed that the adversary only possesses the locked circuit, not the original circuit to query, a prominent attack is applied to these generated netlists collectively, from which a large number of key bits are deciphered. Nevertheless, this paper also describes how the proposed oracle-less attack can be integrated with an oracle-guided attack. The feasibility of the proposed approach is demonstrated for several benchmarks, including remarkable results for breaking a recently proposed provably secure logic locking method and deciphering values of a large number of key bits of the CSAW'19 circuits with very high accuracy.



## **11. Towards Backdoor Attacks and Defense in Robust Machine Learning Models**

cs.CV

Accepted in Computers & Security, 2023

**SubmitDate**: 2023-01-11    [abs](http://arxiv.org/abs/2003.00865v4) [paper-pdf](http://arxiv.org/pdf/2003.00865v4)

**Authors**: Ezekiel Soremekun, Sakshi Udeshi, Sudipta Chattopadhyay

**Abstract**: The introduction of robust optimisation has pushed the state-of-the-art in defending against adversarial attacks. Notably, the state-of-the-art projected gradient descent (PGD)-based training method has been shown to be universally and reliably effective in defending against adversarial inputs. This robustness approach uses PGD as a reliable and universal "first-order adversary". However, the behaviour of such optimisation has not been studied in the light of a fundamentally different class of attacks called backdoors. In this paper, we study how to inject and defend against backdoor attacks for robust models trained using PGD-based robust optimisation. We demonstrate that these models are susceptible to backdoor attacks. Subsequently, we observe that backdoors are reflected in the feature representation of such models. Then, this observation is leveraged to detect such backdoor-infected models via a detection technique called AEGIS. Specifically, given a robust Deep Neural Network (DNN) that is trained using PGD-based first-order adversarial training approach, AEGIS uses feature clustering to effectively detect whether such DNNs are backdoor-infected or clean.   In our evaluation of several visible and hidden backdoor triggers on major classification tasks using CIFAR-10, MNIST and FMNIST datasets, AEGIS effectively detects PGD-trained robust DNNs infected with backdoors. AEGIS detects such backdoor-infected models with 91.6% accuracy (11 out of 12 tested models), without any false positives. Furthermore, AEGIS detects the targeted class in the backdoor-infected model with a reasonably low (11.1%) false positive rate. Our investigation reveals that salient features of adversarially robust DNNs could be promising to break the stealthy nature of backdoor attacks.



## **12. SoK: Adversarial Machine Learning Attacks and Defences in Multi-Agent Reinforcement Learning**

cs.LG

**SubmitDate**: 2023-01-11    [abs](http://arxiv.org/abs/2301.04299v1) [paper-pdf](http://arxiv.org/pdf/2301.04299v1)

**Authors**: Maxwell Standen, Junae Kim, Claudia Szabo

**Abstract**: Multi-Agent Reinforcement Learning (MARL) is vulnerable to Adversarial Machine Learning (AML) attacks and needs adequate defences before it can be used in real world applications. We have conducted a survey into the use of execution-time AML attacks against MARL and the defences against those attacks. We surveyed related work in the application of AML in Deep Reinforcement Learning (DRL) and Multi-Agent Learning (MAL) to inform our analysis of AML for MARL. We propose a novel perspective to understand the manner of perpetrating an AML attack, by defining Attack Vectors. We develop two new frameworks to address a gap in current modelling frameworks, focusing on the means and tempo of an AML attack against MARL, and identify knowledge gaps and future avenues of research.



## **13. User-Centered Security in Natural Language Processing**

cs.CL

PhD thesis, ISBN 978-94-6458-867-5

**SubmitDate**: 2023-01-10    [abs](http://arxiv.org/abs/2301.04230v1) [paper-pdf](http://arxiv.org/pdf/2301.04230v1)

**Authors**: Chris Emmery

**Abstract**: This dissertation proposes a framework of user-centered security in Natural Language Processing (NLP), and demonstrates how it can improve the accessibility of related research. Accordingly, it focuses on two security domains within NLP with great public interest. First, that of author profiling, which can be employed to compromise online privacy through invasive inferences. Without access and detailed insight into these models' predictions, there is no reasonable heuristic by which Internet users might defend themselves from such inferences. Secondly, that of cyberbullying detection, which by default presupposes a centralized implementation; i.e., content moderation across social platforms. As access to appropriate data is restricted, and the nature of the task rapidly evolves (both through lexical variation, and cultural shifts), the effectiveness of its classifiers is greatly diminished and thereby often misrepresented.   Under the proposed framework, we predominantly investigate the use of adversarial attacks on language; i.e., changing a given input (generating adversarial samples) such that a given model does not function as intended. These attacks form a common thread between our user-centered security problems; they are highly relevant for privacy-preserving obfuscation methods against author profiling, and adversarial samples might also prove useful to assess the influence of lexical variation and augmentation on cyberbullying detection.



## **14. AdvBiom: Adversarial Attacks on Biometric Matchers**

cs.CV

arXiv admin note: text overlap with arXiv:1908.05008

**SubmitDate**: 2023-01-10    [abs](http://arxiv.org/abs/2301.03966v1) [paper-pdf](http://arxiv.org/pdf/2301.03966v1)

**Authors**: Debayan Deb, Vishesh Mistry, Rahul Parthe

**Abstract**: With the advent of deep learning models, face recognition systems have achieved impressive recognition rates. The workhorses behind this success are Convolutional Neural Networks (CNNs) and the availability of large training datasets. However, we show that small human-imperceptible changes to face samples can evade most prevailing face recognition systems. Even more alarming is the fact that the same generator can be extended to other traits in the future. In this work, we present how such a generator can be trained and also extended to other biometric modalities, such as fingerprint recognition systems.



## **15. Learned Systems Security**

cs.CR

**SubmitDate**: 2023-01-10    [abs](http://arxiv.org/abs/2212.10318v3) [paper-pdf](http://arxiv.org/pdf/2212.10318v3)

**Authors**: Roei Schuster, Jin Peng Zhou, Thorsten Eisenhofer, Paul Grubbs, Nicolas Papernot

**Abstract**: A learned system uses machine learning (ML) internally to improve performance. We can expect such systems to be vulnerable to some adversarial-ML attacks. Often, the learned component is shared between mutually-distrusting users or processes, much like microarchitectural resources such as caches, potentially giving rise to highly-realistic attacker models. However, compared to attacks on other ML-based systems, attackers face a level of indirection as they cannot interact directly with the learned model. Additionally, the difference between the attack surface of learned and non-learned versions of the same system is often subtle. These factors obfuscate the de-facto risks that the incorporation of ML carries. We analyze the root causes of potentially-increased attack surface in learned systems and develop a framework for identifying vulnerabilities that stem from the use of ML. We apply our framework to a broad set of learned systems under active development. To empirically validate the many vulnerabilities surfaced by our framework, we choose 3 of them and implement and evaluate exploits against prominent learned-system instances. We show that the use of ML caused leakage of past queries in a database, enabled a poisoning attack that causes exponential memory blowup in an index structure and crashes it in seconds, and enabled index users to snoop on each others' key distributions by timing queries over their own keys. We find that adversarial ML is a universal threat against learned systems, point to open research gaps in our understanding of learned-systems security, and conclude by discussing mitigations, while noting that data leakage is inherent in systems whose learned component is shared between multiple parties.



## **16. Over-The-Air Adversarial Attacks on Deep Learning Wi-Fi Fingerprinting**

cs.CR

To appear in the IEEE Internet of Things Journal

**SubmitDate**: 2023-01-10    [abs](http://arxiv.org/abs/2301.03760v1) [paper-pdf](http://arxiv.org/pdf/2301.03760v1)

**Authors**: Fei Xiao, Yong Huang, Yingying Zuo, Wei Kuang, Wei Wang

**Abstract**: Empowered by deep neural networks (DNNs), Wi-Fi fingerprinting has recently achieved astonishing localization performance to facilitate many security-critical applications in wireless networks, but it is inevitably exposed to adversarial attacks, where subtle perturbations can mislead DNNs to wrong predictions. Such vulnerability provides new security breaches to malicious devices for hampering wireless network security, such as malfunctioning geofencing or asset management. The prior adversarial attack on localization DNNs uses additive perturbations on channel state information (CSI) measurements, which is impractical in Wi-Fi transmissions. To transcend this limitation, this paper presents FooLoc, which fools Wi-Fi CSI fingerprinting DNNs over the realistic wireless channel between the attacker and the victim access point (AP). We observe that though uplink CSIs are unknown to the attacker, the accessible downlink CSIs could be their reasonable substitutes at the same spot. We thoroughly investigate the multiplicative and repetitive properties of over-the-air perturbations and devise an efficient optimization problem to generate imperceptible yet robust adversarial perturbations. We implement FooLoc using commercial Wi-Fi APs and Wireless Open-Access Research Platform (WARP) v3 boards in offline and online experiments, respectively. The experimental results show that FooLoc achieves overall attack success rates of about 70% in targeted attacks and of above 90% in untargeted attacks with small perturbation-to-signal ratios of about -18dB.



## **17. On the Susceptibility and Robustness of Time Series Models through Adversarial Attack and Defense**

cs.LG

8 pages, 3 figures, 7 tables

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2301.03703v1) [paper-pdf](http://arxiv.org/pdf/2301.03703v1)

**Authors**: Asadullah Hill Galib, Bidhan Bashyal

**Abstract**: Under adversarial attacks, time series regression and classification are vulnerable. Adversarial defense, on the other hand, can make the models more resilient. It is important to evaluate how vulnerable different time series models are to attacks and how well they recover using defense. The sensitivity to various attacks and the robustness using the defense of several time series models are investigated in this study. Experiments are run on seven-time series models with three adversarial attacks and one adversarial defense. According to the findings, all models, particularly GRU and RNN, appear to be vulnerable. LSTM and GRU also have better defense recovery. FGSM exceeds the competitors in terms of attacks. PGD attacks are more difficult to recover from than other sorts of attacks.



## **18. Adversarial Policies Beat Superhuman Go AIs**

cs.LG

36 pages, 19 figures, see paper for changelog

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2211.00241v2) [paper-pdf](http://arxiv.org/pdf/2211.00241v2)

**Authors**: Tony Tong Wang, Adam Gleave, Nora Belrose, Tom Tseng, Joseph Miller, Kellin Pelrine, Michael D Dennis, Yawen Duan, Viktor Pogrebniak, Sergey Levine, Stuart Russell

**Abstract**: We attack the state-of-the-art Go-playing AI system, KataGo, by training adversarial policies that play against frozen KataGo victims. Our attack achieves a >99% win rate when KataGo uses no tree-search, and a >77% win rate when KataGo uses enough search to be superhuman. Notably, our adversaries do not win by learning to play Go better than KataGo -- in fact, our adversaries are easily beaten by human amateurs. Instead, our adversaries win by tricking KataGo into making serious blunders. Our results demonstrate that even superhuman AI systems may harbor surprising failure modes. Example games are available at https://goattack.far.ai/.



## **19. F3B: A Low-Overhead Blockchain Architecture with Per-Transaction Front-Running Protection**

cs.CR

25 pages, 6 figures

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2205.08529v2) [paper-pdf](http://arxiv.org/pdf/2205.08529v2)

**Authors**: Haoqian Zhang, Louis-Henri Merino, Mahsa Bastankhah, Vero Estrada-Galinanes, Bryan Ford

**Abstract**: Front-running attacks, which benefit from advanced knowledge of pending transactions, have proliferated in the blockchain space, since the emergence of decentralized finance. Front-running causes devastating losses to honest participants and continues to endanger the fairness of the ecosystem. We present Flash Freezing Flash Boys (F3B), a blockchain architecture that addresses front-running attacks by using threshold cryptography. In F3B, a user generates a symmetric key to encrypt their transaction, and once the underlying consensus layer has committed the transaction, a decentralized secret-management committee reveals this key. F3B mitigates front-running attacks because, before the consensus group commits it, an adversary can no longer read the content of a transaction, thus preventing the adversary from benefiting from advanced knowledge of pending transactions. Unlike other threshold-based approaches, where users encrypt their transactions with a key derived from a future block, F3B enables users to generate a unique key for each transaction. This feature ensures that all uncommitted transactions remain private, even if they are delayed. Furthermore, F3B addresses front-running at the execution layer; thus, our solution is agnostic to the underlying consensus algorithm and compatible with existing smart contracts. We evaluated F3B based on Ethereum, demonstrating a 0.05% transaction latency overhead with a secret-management committee of 128 members, thus indicating our solution is practical at a low cost.



## **20. Distributed Estimation over Directed Graphs Resilient to Sensor Spoofing**

eess.SY

12 pages, 9 figures

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2104.04680v2) [paper-pdf](http://arxiv.org/pdf/2104.04680v2)

**Authors**: Shamik Bhattacharyya, Kiran Rokade, Rachel Kalpana Kalaimani

**Abstract**: This paper addresses the problem of distributed estimation of an unknown dynamic parameter by a multi-agent system over a directed communication network in the presence of an adversarial attack on the agents' sensors. The mode of attack of the adversaries is to corrupt the sensor measurements of some of the agents, while the communication and information processing capabilities of those agents remain unaffected. To ensure that all the agents, both normal as well as those under attack, are able to correctly estimate the parameter value, the Resilient Estimation through Weight Balancing (REWB) algorithm is introduced. The only condition required for the REWB algorithm to guarantee resilient estimation is that at any given point in time, less than half of the total number of agents are under attack. The paper discusses the development of the REWB algorithm using the concepts of weight balancing of directed graphs, and the consensus+innovations approach for linear estimation. Numerical simulations are presented to illustrate the performance of our algorithm over directed graphs under different conditions of adversarial attacks.



## **21. Structural Equivalence in Subgraph Matching**

cs.DS

To appear in IEEE Transactions on Network Science and Engineering

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2301.03161v1) [paper-pdf](http://arxiv.org/pdf/2301.03161v1)

**Authors**: Dominic Yang, Yurun Ge, Thien Nguyen, Jacob Moorman, Denali Molitor, Andrea Bertozzi

**Abstract**: Symmetry plays a major role in subgraph matching both in the description of the graphs in question and in how it confounds the search process. This work addresses how to quantify these effects and how to use symmetries to increase the efficiency of subgraph isomorphism algorithms. We introduce rigorous definitions of structural equivalence and establish conditions for when it can be safely used to generate more solutions. We illustrate how to adapt standard search routines to utilize these symmetries to accelerate search and compactly describe the solution space. We then adapt a state-of-the-art solver and perform a comprehensive series of tests to demonstrate these methods' efficacy on a standard benchmark set. We extend these methods to multiplex graphs and present results on large multiplex networks drawn from transportation systems, social media, adversarial attacks, and knowledge graphs.



## **22. RobArch: Designing Robust Architectures against Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2301.03110v1) [paper-pdf](http://arxiv.org/pdf/2301.03110v1)

**Authors**: ShengYun Peng, Weilin Xu, Cory Cornelius, Kevin Li, Rahul Duggal, Duen Horng Chau, Jason Martin

**Abstract**: Adversarial Training is the most effective approach for improving the robustness of Deep Neural Networks (DNNs). However, compared to the large body of research in optimizing the adversarial training process, there are few investigations into how architecture components affect robustness, and they rarely constrain model capacity. Thus, it is unclear where robustness precisely comes from. In this work, we present the first large-scale systematic study on the robustness of DNN architecture components under fixed parameter budgets. Through our investigation, we distill 18 actionable robust network design guidelines that empower model developers to gain deep insights. We demonstrate these guidelines' effectiveness by introducing the novel Robust Architecture (RobArch) model that instantiates the guidelines to build a family of top-performing models across parameter capacities against strong adversarial attacks. RobArch achieves the new state-of-the-art AutoAttack accuracy on the RobustBench ImageNet leaderboard. The code is available at $\href{https://github.com/ShengYun-Peng/RobArch}{\text{this url}}$.



## **23. A Bayesian Robust Regression Method for Corrupted Data Reconstruction**

cs.LG

22 pages

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2212.12787v2) [paper-pdf](http://arxiv.org/pdf/2212.12787v2)

**Authors**: Zheyi Fan, Zhaohui Li, Jingyan Wang, Dennis K. J. Lin, Xiao Xiong, Qingpei Hu

**Abstract**: Because of the widespread existence of noise and data corruption, recovering the true regression parameters with a certain proportion of corrupted response variables is an essential task. Methods to overcome this problem often involve robust least-squares regression, but few methods perform well when confronted with severe adaptive adversarial attacks. In many applications, prior knowledge is often available from historical data or engineering experience, and by incorporating prior information into a robust regression method, we develop an effective robust regression method that can resist adaptive adversarial attacks. First, we propose the novel TRIP (hard Thresholding approach to Robust regression with sImple Prior) algorithm, which improves the breakdown point when facing adaptive adversarial attacks. Then, to improve the robustness and reduce the estimation error caused by the inclusion of priors, we use the idea of Bayesian reweighting to construct the more robust BRHT (robust Bayesian Reweighting regression via Hard Thresholding) algorithm. We prove the theoretical convergence of the proposed algorithms under mild conditions, and extensive experiments show that under different types of dataset attacks, our algorithms outperform other benchmark ones. Finally, we apply our methods to a data-recovery problem in a real-world application involving a space solar array, demonstrating their good applicability.



## **24. Deepfake CAPTCHA: A Method for Preventing Fake Calls**

cs.CR

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2301.03064v1) [paper-pdf](http://arxiv.org/pdf/2301.03064v1)

**Authors**: Lior Yasur, Guy Frankovits, Fred M. Grabovski, Yisroel Mirsky

**Abstract**: Deep learning technology has made it possible to generate realistic content of specific individuals. These `deepfakes' can now be generated in real-time which enables attackers to impersonate people over audio and video calls. Moreover, some methods only need a few images or seconds of audio to steal an identity. Existing defenses perform passive analysis to detect fake content. However, with the rapid progress of deepfake quality, this may be a losing game.   In this paper, we propose D-CAPTCHA: an active defense against real-time deepfakes. The approach is to force the adversary into the spotlight by challenging the deepfake model to generate content which exceeds its capabilities. By doing so, passive detection becomes easier since the content will be distorted. In contrast to existing CAPTCHAs, we challenge the AI's ability to create content as opposed to its ability to classify content. In this work we focus on real-time audio deepfakes and present preliminary results on video.   In our evaluation we found that D-CAPTCHA outperforms state-of-the-art audio deepfake detectors with an accuracy of 91-100% depending on the challenge (compared to 71% without challenges). We also performed a study on 41 volunteers to understand how threatening current real-time deepfake attacks are. We found that the majority of the volunteers could not tell the difference between real and fake audio.



## **25. Byzantine Multiple Access Channels -- Part I: Reliable Communication**

cs.IT

This supercedes Part I of arxiv:1904.11925

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2211.12769v2) [paper-pdf](http://arxiv.org/pdf/2211.12769v2)

**Authors**: Neha Sangwan, Mayank Bakshi, Bikash Kumar Dey, Vinod M. Prabhakaran

**Abstract**: We study communication over a Multiple Access Channel (MAC) where users can possibly be adversarial. The receiver is unaware of the identity of the adversarial users (if any). When all users are non-adversarial, we want their messages to be decoded reliably. When a user behaves adversarially, we require that the honest users' messages be decoded reliably. An adversarial user can mount an attack by sending any input into the channel rather than following the protocol. It turns out that the $2$-user MAC capacity region follows from the point-to-point Arbitrarily Varying Channel (AVC) capacity. For the $3$-user MAC in which at most one user may be malicious, we characterize the capacity region for deterministic codes and randomized codes (where each user shares an independent random secret key with the receiver). These results are then generalized for the $k$-user MAC where the adversary may control all users in one out of a collection of given subsets.



## **26. GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks**

cs.LG

Published as a conference paper at LoG 2022

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2201.12741v6) [paper-pdf](http://arxiv.org/pdf/2201.12741v6)

**Authors**: Chenhui Deng, Xiuyu Li, Zhuo Feng, Zhiru Zhang

**Abstract**: Graph neural networks (GNNs) have been increasingly deployed in various applications that involve learning on non-Euclidean data. However, recent studies show that GNNs are vulnerable to graph adversarial attacks. Although there are several defense methods to improve GNN robustness by eliminating adversarial components, they may also impair the underlying clean graph structure that contributes to GNN training. In addition, few of those defense models can scale to large graphs due to their high computational complexity and memory usage. In this paper, we propose GARNET, a scalable spectral method to boost the adversarial robustness of GNN models. GARNET first leverages weighted spectral embedding to construct a base graph, which is not only resistant to adversarial attacks but also contains critical (clean) graph structure for GNN training. Next, GARNET further refines the base graph by pruning additional uncritical edges based on probabilistic graphical model. GARNET has been evaluated on various datasets, including a large graph with millions of nodes. Our extensive experiment results show that GARNET achieves adversarial accuracy improvement and runtime speedup over state-of-the-art GNN (defense) models by up to 13.27% and 14.7x, respectively.



## **27. Robust Feature-Level Adversaries are Interpretability Tools**

cs.LG

Code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2023-01-07    [abs](http://arxiv.org/abs/2110.03605v6) [paper-pdf](http://arxiv.org/pdf/2110.03605v6)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstract**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore perceptible, interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying representations in models. Second, we show that these adversaries are uniquely versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results suggest that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations. Code is available at https://github.com/thestephencasper/feature_level_adv



## **28. Adversarial training with informed data selection**

cs.LG

**SubmitDate**: 2023-01-07    [abs](http://arxiv.org/abs/2301.04472v1) [paper-pdf](http://arxiv.org/pdf/2301.04472v1)

**Authors**: Marcele O. K. Mendonça, Javier Maroto, Pascal Frossard, Paulo S. R. Diniz

**Abstract**: With the increasing amount of available data and advances in computing capabilities, deep neural networks (DNNs) have been successfully employed to solve challenging tasks in various areas, including healthcare, climate, and finance. Nevertheless, state-of-the-art DNNs are susceptible to quasi-imperceptible perturbed versions of the original images -- adversarial examples. These perturbations of the network input can lead to disastrous implications in critical areas where wrong decisions can directly affect human lives. Adversarial training is the most efficient solution to defend the network against these malicious attacks. However, adversarial trained networks generally come with lower clean accuracy and higher computational complexity. This work proposes a data selection (DS) strategy to be applied in the mini-batch training. Based on the cross-entropy loss, the most relevant samples in the batch are selected to update the model parameters in the backpropagation. The simulation results show that a good compromise can be obtained regarding robustness and standard accuracy, whereas the computational complexity of the backpropagation pass is reduced.



## **29. PatchUp: A Feature-Space Block-Level Regularization Technique for Convolutional Neural Networks**

cs.LG

AAAI - 2022

**SubmitDate**: 2023-01-07    [abs](http://arxiv.org/abs/2006.07794v2) [paper-pdf](http://arxiv.org/pdf/2006.07794v2)

**Authors**: Mojtaba Faramarzi, Mohammad Amini, Akilesh Badrinaaraayanan, Vikas Verma, Sarath Chandar

**Abstract**: Large capacity deep learning models are often prone to a high generalization gap when trained with a limited amount of labeled training data. A recent class of methods to address this problem uses various ways to construct a new training sample by mixing a pair (or more) of training samples. We propose PatchUp, a hidden state block-level regularization technique for Convolutional Neural Networks (CNNs), that is applied on selected contiguous blocks of feature maps from a random pair of samples. Our approach improves the robustness of CNN models against the manifold intrusion problem that may occur in other state-of-the-art mixing approaches. Moreover, since we are mixing the contiguous block of features in the hidden space, which has more dimensions than the input space, we obtain more diverse samples for training towards different dimensions. Our experiments on CIFAR10/100, SVHN, Tiny-ImageNet, and ImageNet using ResNet architectures including PreActResnet18/34, WRN-28-10, ResNet101/152 models show that PatchUp improves upon, or equals, the performance of current state-of-the-art regularizers for CNNs. We also show that PatchUp can provide a better generalization to deformed samples and is more robust against adversarial attacks.



## **30. Stealthy Backdoor Attack for Code Models**

cs.CR

Under review of IEEE Transactions on Software Engineering

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02496v1) [paper-pdf](http://arxiv.org/pdf/2301.02496v1)

**Authors**: Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo

**Abstract**: Code models, such as CodeBERT and CodeT5, offer general-purpose representations of code and play a vital role in supporting downstream automated software engineering tasks. Most recently, code models were revealed to be vulnerable to backdoor attacks. A code model that is backdoor-attacked can behave normally on clean examples but will produce pre-defined malicious outputs on examples injected with triggers that activate the backdoors. Existing backdoor attacks on code models use unstealthy and easy-to-detect triggers. This paper aims to investigate the vulnerability of code models with stealthy backdoor attacks. To this end, we propose AFRAIDOOR (Adversarial Feature as Adaptive Backdoor). AFRAIDOOR achieves stealthiness by leveraging adversarial perturbations to inject adaptive triggers into different inputs. We evaluate AFRAIDOOR on three widely adopted code models (CodeBERT, PLBART and CodeT5) and two downstream tasks (code summarization and method name prediction). We find that around 85% of adaptive triggers in AFRAIDOOR bypass the detection in the defense process. By contrast, only less than 12% of the triggers from previous work bypass the defense. When the defense method is not applied, both AFRAIDOOR and baselines have almost perfect attack success rates. However, once a defense is applied, the success rates of baselines decrease dramatically to 10.47% and 12.06%, while the success rate of AFRAIDOOR are 77.05% and 92.98% on the two tasks. Our finding exposes security weaknesses in code models under stealthy backdoor attacks and shows that the state-of-the-art defense method cannot provide sufficient protection. We call for more research efforts in understanding security threats to code models and developing more effective countermeasures.



## **31. Watching your call: Breaking VoLTE Privacy in LTE/5G Networks**

cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02487v1) [paper-pdf](http://arxiv.org/pdf/2301.02487v1)

**Authors**: Zishuai Cheng, Mihai Ordean, Flavio D. Garcia, Baojiang Cui, Dominik Rys

**Abstract**: Voice over LTE (VoLTE) and Voice over NR (VoNR) are two similar technologies that have been widely deployed by operators to provide a better calling experience in LTE and 5G networks, respectively. The VoLTE/NR protocols rely on the security features of the underlying LTE/5G network to protect users' privacy such that nobody can monitor calls and learn details about call times, duration, and direction. In this paper, we introduce a new privacy attack which enables adversaries to analyse encrypted LTE/5G traffic and recover any VoLTE/NR call details. We achieve this by implementing a novel mobile-relay adversary which is able to remain undetected by using an improved physical layer parameter guessing procedure. This adversary facilitates the recovery of encrypted configuration messages exchanged between victim devices and the mobile network. We further propose an identity mapping method which enables our mobile-relay adversary to link a victim's network identifiers to the phone number efficiently, requiring a single VoLTE protocol message. We evaluate the real-world performance of our attacks using four modern commercial off-the-shelf phones and two representative, commercial network carriers. We collect over 60 hours of traffic between the phones and the mobile networks and execute 160 VoLTE calls, which we use to successfully identify patterns in the physical layer parameter allocation and in VoLTE traffic, respectively. Our real-world experiments show that our mobile-relay works as expected in all test cases, and the VoLTE activity logs recovered describe the actual communication with 100% accuracy. Finally, we show that we can link network identifiers such as International Mobile Subscriber Identities (IMSI), Subscriber Concealed Identifiers (SUCI) and/or Globally Unique Temporary Identifiers (GUTI) to phone numbers while remaining undetected by the victim.



## **32. Adversarial Attacks on Neural Models of Code via Code Difference Reduction**

cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02412v1) [paper-pdf](http://arxiv.org/pdf/2301.02412v1)

**Authors**: Zhao Tian, Junjie Chen, Zhi Jin

**Abstract**: Deep learning has been widely used to solve various code-based tasks by building deep code models based on a large number of code snippets. However, deep code models are still vulnerable to adversarial attacks. As source code is discrete and has to strictly stick to the grammar and semantics constraints, the adversarial attack techniques in other domains are not applicable. Moreover, the attack techniques specific to deep code models suffer from the effectiveness issue due to the enormous attack space. In this work, we propose a novel adversarial attack technique (i.e., CODA). Its key idea is to use the code differences between the target input and reference inputs (that have small code differences but different prediction results with the target one) to guide the generation of adversarial examples. It considers both structure differences and identifier differences to preserve the original semantics. Hence, the attack space can be largely reduced as the one constituted by the two kinds of code differences, and thus the attack process can be largely improved by designing corresponding equivalent structure transformations and identifier renaming transformations. Our experiments on 10 deep code models (i.e., two pre trained models with five code-based tasks) demonstrate the effectiveness and efficiency of CODA, the naturalness of its generated examples, and its capability of defending against attacks after adversarial fine-tuning. For example, CODA improves the state-of-the-art techniques (i.e., CARROT and ALERT) by 79.25% and 72.20% on average in terms of the attack success rate, respectively.



## **33. TrojanPuzzle: Covertly Poisoning Code-Suggestion Models**

cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02344v1) [paper-pdf](http://arxiv.org/pdf/2301.02344v1)

**Authors**: Hojjat Aghakhani, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, Robert Sim

**Abstract**: With tools like GitHub Copilot, automatic code suggestion is no longer a dream in software engineering. These tools, based on large language models, are typically trained on massive corpora of code mined from unvetted public sources. As a result, these models are susceptible to data poisoning attacks where an adversary manipulates the model's training or fine-tuning phases by injecting malicious data. Poisoning attacks could be designed to influence the model's suggestions at run time for chosen contexts, such as inducing the model into suggesting insecure code payloads. To achieve this, prior poisoning attacks explicitly inject the insecure code payload into the training data, making the poisoning data detectable by static analysis tools that can remove such malicious data from the training set. In this work, we demonstrate two novel data poisoning attacks, COVERT and TROJANPUZZLE, that can bypass static analysis by planting malicious poisoning data in out-of-context regions such as docstrings. Our most novel attack, TROJANPUZZLE, goes one step further in generating less suspicious poisoning data by never including certain (suspicious) parts of the payload in the poisoned data, while still inducing a model that suggests the entire payload when completing code (i.e., outside docstrings). This makes TROJANPUZZLE robust against signature-based dataset-cleansing methods that identify and filter out suspicious sequences from the training data. Our evaluation against two model sizes demonstrates that both COVERT and TROJANPUZZLE have significant implications for how practitioners should select code used to train or tune code-suggestion models.



## **34. DRL-GAN: A Hybrid Approach for Binary and Multiclass Network Intrusion Detection**

cs.CR

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.03368v1) [paper-pdf](http://arxiv.org/pdf/2301.03368v1)

**Authors**: Caroline Strickland, Chandrika Saha, Muhammad Zakar, Sareh Nejad, Noshin Tasnim, Daniel Lizotte, Anwar Haque

**Abstract**: Our increasingly connected world continues to face an ever-growing amount of network-based attacks. Intrusion detection systems (IDS) are an essential security technology for detecting these attacks. Although numerous machine learning-based IDS have been proposed for the detection of malicious network traffic, the majority have difficulty properly detecting and classifying the more uncommon attack types. In this paper, we implement a novel hybrid technique using synthetic data produced by a Generative Adversarial Network (GAN) to use as input for training a Deep Reinforcement Learning (DRL) model. Our GAN model is trained with the NSL-KDD dataset for four attack categories as well as normal network flow. Ultimately, our findings demonstrate that training the DRL on specific synthetic datasets can result in better performance in correctly classifying minority classes over training on the true imbalanced dataset.



## **35. Silent Killer: Optimizing Backdoor Trigger Yields a Stealthy and Powerful Data Poisoning Attack**

cs.CR

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.02615v1) [paper-pdf](http://arxiv.org/pdf/2301.02615v1)

**Authors**: Tzvi Lederer, Gallil Maimon, Lior Rokach

**Abstract**: We propose a stealthy and powerful backdoor attack on neural networks based on data poisoning (DP). In contrast to previous attacks, both the poison and the trigger in our method are stealthy. We are able to change the model's classification of samples from a source class to a target class chosen by the attacker. We do so by using a small number of poisoned training samples with nearly imperceptible perturbations, without changing their labels. At inference time, we use a stealthy perturbation added to the attacked samples as a trigger. This perturbation is crafted as a universal adversarial perturbation (UAP), and the poison is crafted using gradient alignment coupled to this trigger. Our method is highly efficient in crafting time compared to previous methods and requires only a trained surrogate model without additional retraining. Our attack achieves state-of-the-art results in terms of attack success rate while maintaining high accuracy on clean samples.



## **36. Holistic Adversarial Robustness of Deep Learning Models**

cs.LG

survey paper on holistic adversarial robustness for deep learning;  published at AAAI 2023 Senior Member Presentation Track

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2202.07201v3) [paper-pdf](http://arxiv.org/pdf/2202.07201v3)

**Authors**: Pin-Yu Chen, Sijia Liu

**Abstract**: Adversarial robustness studies the worst-case performance of a machine learning model to ensure safety and reliability. With the proliferation of deep-learning-based technology, the potential risks associated with model development and deployment can be amplified and become dreadful vulnerabilities. This paper provides a comprehensive overview of research topics and foundational principles of research methods for adversarial robustness of deep learning models, including attacks, defenses, verification, and novel applications.



## **37. Randomized Message-Interception Smoothing: Gray-box Certificates for Graph Neural Networks**

cs.LG

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.02039v1) [paper-pdf](http://arxiv.org/pdf/2301.02039v1)

**Authors**: Yan Scholten, Jan Schuchardt, Simon Geisler, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Randomized smoothing is one of the most promising frameworks for certifying the adversarial robustness of machine learning models, including Graph Neural Networks (GNNs). Yet, existing randomized smoothing certificates for GNNs are overly pessimistic since they treat the model as a black box, ignoring the underlying architecture. To remedy this, we propose novel gray-box certificates that exploit the message-passing principle of GNNs: We randomly intercept messages and carefully analyze the probability that messages from adversarially controlled nodes reach their target nodes. Compared to existing certificates, we certify robustness to much stronger adversaries that control entire nodes in the graph and can arbitrarily manipulate node features. Our certificates provide stronger guarantees for attacks at larger distances, as messages from farther-away nodes are more likely to get intercepted. We demonstrate the effectiveness of our method on various models and datasets. Since our gray-box certificates consider the underlying graph structure, we can significantly improve certifiable robustness by applying graph sparsification.



## **38. Beckman Defense**

cs.LG

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.01495v2) [paper-pdf](http://arxiv.org/pdf/2301.01495v2)

**Authors**: A. V. Subramanyam

**Abstract**: Optimal transport (OT) based distributional robust optimisation (DRO) has received some traction in the recent past. However, it is at a nascent stage but has a sound potential in robustifying the deep learning models. Interestingly, OT barycenters demonstrate a good robustness against adversarial attacks. Owing to the computationally expensive nature of OT barycenters, they have not been investigated under DRO framework. In this work, we propose a new barycenter, namely Beckman barycenter, which can be computed efficiently and used for training the network to defend against adversarial attacks in conjunction with adversarial training. We propose a novel formulation of Beckman barycenter and analytically obtain the barycenter using the marginals of the input image. We show that the Beckman barycenter can be used to train adversarially trained networks to improve the robustness. Our training is extremely efficient as it requires only a single epoch of training. Elaborate experiments on CIFAR-10, CIFAR-100 and Tiny ImageNet demonstrate that training an adversarially robust network with Beckman barycenter can significantly increase the performance. Under auto attack, we get a a maximum boost of 10\% in CIFAR-10, 8.34\% in CIFAR-100 and 11.51\% in Tiny ImageNet. Our code is available at https://github.com/Visual-Conception-Group/test-barycentric-defense.



## **39. Enhancement attacks in biomedical machine learning**

stat.ML

13 pages, 3 figures

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.01885v1) [paper-pdf](http://arxiv.org/pdf/2301.01885v1)

**Authors**: Matthew Rosenblatt, Javid Dadashkarimi, Dustin Scheinost

**Abstract**: The prevalence of machine learning in biomedical research is rapidly growing, yet the trustworthiness of such research is often overlooked. While some previous works have investigated the ability of adversarial attacks to degrade model performance in medical imaging, the ability to falsely improve performance via recently-developed "enhancement attacks" may be a greater threat to biomedical machine learning. In the spirit of developing attacks to better understand trustworthiness, we developed three techniques to drastically enhance prediction performance of classifiers with minimal changes to features, including the enhancement of 1) within-dataset predictions, 2) a particular method over another, and 3) cross-dataset generalization. Our within-dataset enhancement framework falsely improved classifiers' accuracy from 50% to almost 100% while maintaining high feature similarities between original and enhanced data (Pearson's r's>0.99). Similarly, the method-specific enhancement framework was effective in falsely improving the performance of one method over another. For example, a simple neural network outperformed LR by 50% on our enhanced dataset, although no performance differences were present in the original dataset. Crucially, the original and enhanced data were still similar (r=0.95). Finally, we demonstrated that enhancement is not specific to within-dataset predictions but can also be adapted to enhance the generalization accuracy of one dataset to another by up to 38%. Overall, our results suggest that more robust data sharing and provenance tracking pipelines are necessary to maintain data integrity in biomedical machine learning research.



## **40. Availability Adversarial Attack and Countermeasures for Deep Learning-based Load Forecasting**

cs.LG

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01832v1) [paper-pdf](http://arxiv.org/pdf/2301.01832v1)

**Authors**: Wangkun Xu, Fei Teng

**Abstract**: The forecast of electrical loads is essential for the planning and operation of the power system. Recently, advances in deep learning have enabled more accurate forecasts. However, deep neural networks are prone to adversarial attacks. Although most of the literature focuses on integrity-based attacks, this paper proposes availability-based adversarial attacks, which can be more easily implemented by attackers. For each forecast instance, the availability attack position is optimally solved by mixed-integer reformulation of the artificial neural network. To tackle this attack, an adversarial training algorithm is proposed. In simulation, a realistic load forecasting dataset is considered and the attack performance is compared to the integrity-based attack. Meanwhile, the adversarial training algorithm is shown to significantly improve robustness against availability attacks. All codes are available at https://github.com/xuwkk/AAA_Load_Forecast.



## **41. GUAP: Graph Universal Attack Through Adversarial Patching**

cs.LG

8 pages

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01731v1) [paper-pdf](http://arxiv.org/pdf/2301.01731v1)

**Authors**: Xiao Zang, Jie Chen, Bo Yuan

**Abstract**: Graph neural networks (GNNs) are a class of effective deep learning models for node classification tasks; yet their predictive capability may be severely compromised under adversarially designed unnoticeable perturbations to the graph structure and/or node data. Most of the current work on graph adversarial attacks aims at lowering the overall prediction accuracy, but we argue that the resulting abnormal model performance may catch attention easily and invite quick counterattack. Moreover, attacks through modification of existing graph data may be hard to conduct if good security protocols are implemented. In this work, we consider an easier attack harder to be noticed, through adversarially patching the graph with new nodes and edges. The attack is universal: it targets a single node each time and flips its connection to the same set of patch nodes. The attack is unnoticeable: it does not modify the predictions of nodes other than the target. We develop an algorithm, named GUAP, that achieves high attack success rate but meanwhile preserves the prediction accuracy. GUAP is fast to train by employing a sampling strategy. We demonstrate that a 5% sampling in each epoch yields 20x speedup in training, with only a slight degradation in attack performance. Additionally, we show that the adversarial patch trained with the graph convolutional network transfers well to other GNNs, such as the graph attention network.



## **42. A Survey on Physical Adversarial Attack in Computer Vision**

cs.CV

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2209.14262v2) [paper-pdf](http://arxiv.org/pdf/2209.14262v2)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Guijian Tang, Xiaoqian Chen

**Abstract**: In the past decade, deep learning has dramatically changed the traditional hand-craft feature manner with strong feature learning capability, resulting in tremendous improvement of conventional tasks. However, deep neural networks have recently been demonstrated vulnerable to adversarial examples, a kind of malicious samples crafted by small elaborately designed noise, which mislead the DNNs to make the wrong decisions while remaining imperceptible to humans. Adversarial examples can be divided into digital adversarial attacks and physical adversarial attacks. The digital adversarial attack is mostly performed in lab environments, focusing on improving the performance of adversarial attack algorithms. In contrast, the physical adversarial attack focus on attacking the physical world deployed DNN systems, which is a more challenging task due to the complex physical environment (i.e., brightness, occlusion, and so on). Although the discrepancy between digital adversarial and physical adversarial examples is small, the physical adversarial examples have a specific design to overcome the effect of the complex physical environment. In this paper, we review the development of physical adversarial attacks in DNN-based computer vision tasks, including image recognition tasks, object detection tasks, and semantic segmentation. For the sake of completeness of the algorithm evolution, we will briefly introduce the works that do not involve the physical adversarial attack. We first present a categorization scheme to summarize the current physical adversarial attacks. Then discuss the advantages and disadvantages of the existing physical adversarial attacks and focus on the technique used to maintain the adversarial when applied into physical environment. Finally, we point out the issues of the current physical adversarial attacks to be solved and provide promising research directions.



## **43. Validity in Music Information Research Experiments**

cs.SD

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01578v1) [paper-pdf](http://arxiv.org/pdf/2301.01578v1)

**Authors**: Bob L. T. Sturm, Arthur Flexer

**Abstract**: Validity is the truth of an inference made from evidence, such as data collected in an experiment, and is central to working scientifically. Given the maturity of the domain of music information research (MIR), validity in our opinion should be discussed and considered much more than it has been so far. Considering validity in one's work can improve its scientific and engineering value. Puzzling MIR phenomena like adversarial attacks and performance glass ceilings become less mysterious through the lens of validity. In this article, we review the subject of validity in general, considering the four major types of validity from a key reference: Shadish et al. 2002. We ground our discussion of these types with a prototypical MIR experiment: music classification using machine learning. Through this MIR experimentalists can be guided to make valid inferences from data collected from their experiments.



## **44. Passive Triangulation Attack on ORide**

cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2208.12216v3) [paper-pdf](http://arxiv.org/pdf/2208.12216v3)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.



## **45. The Feasibility and Inevitability of Stealth Attacks**

cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2106.13997v4) [paper-pdf](http://arxiv.org/pdf/2106.13997v4)

**Authors**: Ivan Y. Tyukin, Desmond J. Higham, Alexander Bastounis, Eliyas Woldegeorgis, Alexander N. Gorban

**Abstract**: We develop and study new adversarial perturbations that enable an attacker to gain control over decisions in generic Artificial Intelligence (AI) systems including deep learning neural networks. In contrast to adversarial data modification, the attack mechanism we consider here involves alterations to the AI system itself. Such a stealth attack could be conducted by a mischievous, corrupt or disgruntled member of a software development team. It could also be made by those wishing to exploit a ``democratization of AI'' agenda, where network architectures and trained parameter sets are shared publicly. We develop a range of new implementable attack strategies with accompanying analysis, showing that with high probability a stealth attack can be made transparent, in the sense that system performance is unchanged on a fixed validation set which is unknown to the attacker, while evoking any desired output on a trigger input of interest. The attacker only needs to have estimates of the size of the validation set and the spread of the AI's relevant latent space. In the case of deep learning neural networks, we show that a one neuron attack is possible - a modification to the weights and bias associated with a single neuron - revealing a vulnerability arising from over-parameterization. We illustrate these concepts using state of the art architectures on two standard image data sets. Guided by the theory and computational results, we also propose strategies to guard against stealth attacks.



## **46. Driver Locations Harvesting Attack on pRide**

cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2210.13263v3) [paper-pdf](http://arxiv.org/pdf/2210.13263v3)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.



## **47. Organised Firestorm as strategy for business cyber-attacks**

cs.CY

9 pages, 3 figures, 2 table

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01518v1) [paper-pdf](http://arxiv.org/pdf/2301.01518v1)

**Authors**: Andrea Russo

**Abstract**: Having a good reputation is paramount for most organisations and companies. In fact, having an optimal corporate image allows them to have better transaction relationships with various customers and partners. However, such reputation is hard to build and easy to destroy for all kind of business commercial activities (B2C, B2B, B2B2C, B2G). A misunderstanding during the communication process to the customers, or just a bad communication strategy, can lead to a disaster for the entire company. This is emphasised by the reaction of millions of people on social networks, which can be very detrimental for the corporate image if they react negatively to a certain event. This is called a firestorm.   In this paper, I propose a well-organised strategy for firestorm attacks on organisations, also showing how an adversary can leverage them to obtain private information on the attacked firm. Standard business security procedures are not designed to operate against multi-domain attacks; therefore, I will show how it is possible to bypass the classic and advised security procedures by operating different kinds of attack. I also propose a different firestorm attack, targeting a specific business company network in an efficient way. Finally, I present defensive procedures to reduce the negative effect of firestorms on a company.



## **48. Universal adversarial perturbation for remote sensing images**

cs.CV

Published in the Twenty-Fourth International Workshop on Multimedia  Signal Processing, MMSP 2022

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2202.10693v2) [paper-pdf](http://arxiv.org/pdf/2202.10693v2)

**Authors**: Qingyu Wang, Guorui Feng, Zhaoxia Yin, Bin Luo

**Abstract**: Recently, with the application of deep learning in the remote sensing image (RSI) field, the classification accuracy of the RSI has been dramatically improved compared with traditional technology. However, even the state-of-the-art object recognition convolutional neural networks are fooled by the universal adversarial perturbation (UAP). The research on UAP is mostly limited to ordinary images, and RSIs have not been studied. To explore the basic characteristics of UAPs of RSIs, this paper proposes a novel method combining an encoder-decoder network with an attention mechanism to generate the UAP of RSIs. Firstly, the former is used to generate the UAP, which can learn the distribution of perturbations better, and then the latter is used to find the sensitive regions concerned by the RSI classification model. Finally, the generated regions are used to fine-tune the perturbation making the model misclassified with fewer perturbations. The experimental results show that the UAP can make the classification model misclassify, and the attack success rate of our proposed method on the RSI data set is as high as 97.09%.



## **49. Surveillance Face Anti-spoofing**

cs.CV

15 pages, 9 figures

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2301.00975v1) [paper-pdf](http://arxiv.org/pdf/2301.00975v1)

**Authors**: Hao Fang, Ajian Liu, Jun Wan, Sergio Escalera, Chenxu Zhao, Xu Zhang, Stan Z. Li, Zhen Lei

**Abstract**: Face Anti-spoofing (FAS) is essential to secure face recognition systems from various physical attacks. However, recent research generally focuses on short-distance applications (i.e., phone unlocking) while lacking consideration of long-distance scenes (i.e., surveillance security checks). In order to promote relevant research and fill this gap in the community, we collect a large-scale Surveillance High-Fidelity Mask (SuHiFiMask) dataset captured under 40 surveillance scenes, which has 101 subjects from different age groups with 232 3D attacks (high-fidelity masks), 200 2D attacks (posters, portraits, and screens), and 2 adversarial attacks. In this scene, low image resolution and noise interference are new challenges faced in surveillance FAS. Together with the SuHiFiMask dataset, we propose a Contrastive Quality-Invariance Learning (CQIL) network to alleviate the performance degradation caused by image quality from three aspects: (1) An Image Quality Variable module (IQV) is introduced to recover image information associated with discrimination by combining the super-resolution network. (2) Using generated sample pairs to simulate quality variance distributions to help contrastive learning strategies obtain robust feature representation under quality variation. (3) A Separate Quality Network (SQN) is designed to learn discriminative features independent of image quality. Finally, a large number of experiments verify the quality of the SuHiFiMask dataset and the superiority of the proposed CQIL.



## **50. Efficient Robustness Assessment via Adversarial Spatial-Temporal Focus on Videos**

cs.CV

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2301.00896v1) [paper-pdf](http://arxiv.org/pdf/2301.00896v1)

**Authors**: Wei Xingxing, Wang Songping, Yan Huanqian

**Abstract**: Adversarial robustness assessment for video recognition models has raised concerns owing to their wide applications on safety-critical tasks. Compared with images, videos have much high dimension, which brings huge computational costs when generating adversarial videos. This is especially serious for the query-based black-box attacks where gradient estimation for the threat models is usually utilized, and high dimensions will lead to a large number of queries. To mitigate this issue, we propose to simultaneously eliminate the temporal and spatial redundancy within the video to achieve an effective and efficient gradient estimation on the reduced searching space, and thus query number could decrease. To implement this idea, we design the novel Adversarial spatial-temporal Focus (AstFocus) attack on videos, which performs attacks on the simultaneously focused key frames and key regions from the inter-frames and intra-frames in the video. AstFocus attack is based on the cooperative Multi-Agent Reinforcement Learning (MARL) framework. One agent is responsible for selecting key frames, and another agent is responsible for selecting key regions. These two agents are jointly trained by the common rewards received from the black-box threat models to perform a cooperative prediction. By continuously querying, the reduced searching space composed of key frames and key regions is becoming precise, and the whole query number becomes less than that on the original video. Extensive experiments on four mainstream video recognition models and three widely used action recognition datasets demonstrate that the proposed AstFocus attack outperforms the SOTA methods, which is prevenient in fooling rate, query number, time, and perturbation magnitude at the same.



