# Latest Adversarial Attack Papers
**update at 2024-06-03 11:57:00**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.13401v3) [paper-pdf](http://arxiv.org/pdf/2405.13401v3)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.



## **2. ACE: A Model Poisoning Attack on Contribution Evaluation Methods in Federated Learning**

cs.CR

To appear in the 33rd USENIX Security Symposium, 2024

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20975v1) [paper-pdf](http://arxiv.org/pdf/2405.20975v1)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bo Li, Radha Poovendran

**Abstract**: In Federated Learning (FL), a set of clients collaboratively train a machine learning model (called global model) without sharing their local training data. The local training data of clients is typically non-i.i.d. and heterogeneous, resulting in varying contributions from individual clients to the final performance of the global model. In response, many contribution evaluation methods were proposed, where the server could evaluate the contribution made by each client and incentivize the high-contributing clients to sustain their long-term participation in FL. Existing studies mainly focus on developing new metrics or algorithms to better measure the contribution of each client. However, the security of contribution evaluation methods of FL operating in adversarial environments is largely unexplored. In this paper, we propose the first model poisoning attack on contribution evaluation methods in FL, termed ACE. Specifically, we show that any malicious client utilizing ACE could manipulate the parameters of its local model such that it is evaluated to have a high contribution by the server, even when its local training data is indeed of low quality. We perform both theoretical analysis and empirical evaluations of ACE. Theoretically, we show our design of ACE can effectively boost the malicious client's perceived contribution when the server employs the widely-used cosine distance metric to measure contribution. Empirically, our results show ACE effectively and efficiently deceive five state-of-the-art contribution evaluation methods. In addition, ACE preserves the accuracy of the final global models on testing inputs. We also explore six countermeasures to defend ACE. Our results show they are inadequate to thwart ACE, highlighting the urgent need for new defenses to safeguard the contribution evaluation methods in FL.



## **3. BackdoorIndicator: Leveraging OOD Data for Proactive Backdoor Detection in Federated Learning**

cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20862v1) [paper-pdf](http://arxiv.org/pdf/2405.20862v1)

**Authors**: Songze Li, Yanbo Dai

**Abstract**: In a federated learning (FL) system, decentralized data owners (clients) could upload their locally trained models to a central server, to jointly train a global model. Malicious clients may plant backdoors into the global model through uploading poisoned local models, causing misclassification to a target class when encountering attacker-defined triggers. Existing backdoor defenses show inconsistent performance under different system and adversarial settings, especially when the malicious updates are made statistically close to the benign ones. In this paper, we first reveal the fact that planting subsequent backdoors with the same target label could significantly help to maintain the accuracy of previously planted backdoors, and then propose a novel proactive backdoor detection mechanism for FL named BackdoorIndicator, which has the server inject indicator tasks into the global model leveraging out-of-distribution (OOD) data, and then utilizing the fact that any backdoor samples are OOD samples with respect to benign samples, the server, who is completely agnostic of the potential backdoor types and target labels, can accurately detect the presence of backdoors in uploaded models, via evaluating the indicator tasks. We perform systematic and extensive empirical studies to demonstrate the consistently superior performance and practicality of BackdoorIndicator over baseline defenses, across a wide range of system and adversarial settings.



## **4. GANcrop: A Contrastive Defense Against Backdoor Attacks in Federated Learning**

cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20727v1) [paper-pdf](http://arxiv.org/pdf/2405.20727v1)

**Authors**: Xiaoyun Gan, Shanyu Gan, Taizhi Su, Peng Liu

**Abstract**: With heightened awareness of data privacy protection, Federated Learning (FL) has attracted widespread attention as a privacy-preserving distributed machine learning method. However, the distributed nature of federated learning also provides opportunities for backdoor attacks, where attackers can guide the model to produce incorrect predictions without affecting the global model training process.   This paper introduces a novel defense mechanism against backdoor attacks in federated learning, named GANcrop. This approach leverages contrastive learning to deeply explore the disparities between malicious and benign models for attack identification, followed by the utilization of Generative Adversarial Networks (GAN) to recover backdoor triggers and implement targeted mitigation strategies. Experimental findings demonstrate that GANcrop effectively safeguards against backdoor attacks, particularly in non-IID scenarios, while maintaining satisfactory model accuracy, showcasing its remarkable defensive efficacy and practical utility.



## **5. Robust Stable Spiking Neural Networks**

cs.NE

Accepted by ICML2024

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20694v1) [paper-pdf](http://arxiv.org/pdf/2405.20694v1)

**Authors**: Jianhao Ding, Zhiyu Pan, Yujia Liu, Zhaofei Yu, Tiejun Huang

**Abstract**: Spiking neural networks (SNNs) are gaining popularity in deep learning due to their low energy budget on neuromorphic hardware. However, they still face challenges in lacking sufficient robustness to guard safety-critical applications such as autonomous driving. Many studies have been conducted to defend SNNs from the threat of adversarial attacks. This paper aims to uncover the robustness of SNN through the lens of the stability of nonlinear systems. We are inspired by the fact that searching for parameters altering the leaky integrate-and-fire dynamics can enhance their robustness. Thus, we dive into the dynamics of membrane potential perturbation and simplify the formulation of the dynamics. We present that membrane potential perturbation dynamics can reliably convey the intensity of perturbation. Our theoretical analyses imply that the simplified perturbation dynamics satisfy input-output stability. Thus, we propose a training framework with modified SNN neurons and to reduce the mean square of membrane potential perturbation aiming at enhancing the robustness of SNN. Finally, we experimentally verify the effectiveness of the framework in the setting of Gaussian noise training and adversarial training on the image classification task.



## **6. Investigating and unmasking feature-level vulnerabilities of CNNs to adversarial perturbations**

cs.CV

22 pages, 15 figures (including appendix)

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20672v1) [paper-pdf](http://arxiv.org/pdf/2405.20672v1)

**Authors**: Davide Coppola, Hwee Kuan Lee

**Abstract**: This study explores the impact of adversarial perturbations on Convolutional Neural Networks (CNNs) with the aim of enhancing the understanding of their underlying mechanisms. Despite numerous defense methods proposed in the literature, there is still an incomplete understanding of this phenomenon. Instead of treating the entire model as vulnerable, we propose that specific feature maps learned during training contribute to the overall vulnerability. To investigate how the hidden representations learned by a CNN affect its vulnerability, we introduce the Adversarial Intervention framework. Experiments were conducted on models trained on three well-known computer vision datasets, subjecting them to attacks of different nature. Our focus centers on the effects that adversarial perturbations to a model's initial layer have on the overall behavior of the model. Empirical results revealed compelling insights: a) perturbing selected channel combinations in shallow layers causes significant disruptions; b) the channel combinations most responsible for the disruptions are common among different types of attacks; c) despite shared vulnerable combinations of channels, different attacks affect hidden representations with varying magnitudes; d) there exists a positive correlation between a kernel's magnitude and its vulnerability. In conclusion, this work introduces a novel framework to study the vulnerability of a CNN model to adversarial perturbations, revealing insights that contribute to a deeper understanding of the phenomenon. The identified properties pave the way for the development of efficient ad-hoc defense mechanisms in future applications.



## **7. Query Provenance Analysis for Robust and Efficient Query-based Black-box Attack Defense**

cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20641v1) [paper-pdf](http://arxiv.org/pdf/2405.20641v1)

**Authors**: Shaofei Li, Ziqi Zhang, Haomin Jia, Ding Li, Yao Guo, Xiangqun Chen

**Abstract**: Query-based black-box attacks have emerged as a significant threat to machine learning systems, where adversaries can manipulate the input queries to generate adversarial examples that can cause misclassification of the model. To counter these attacks, researchers have proposed Stateful Defense Models (SDMs) for detecting adversarial query sequences and rejecting queries that are "similar" to the history queries. Existing state-of-the-art (SOTA) SDMs (e.g., BlackLight and PIHA) have shown great effectiveness in defending against these attacks. However, recent studies have shown that they are vulnerable to Oracle-guided Adaptive Rejection Sampling (OARS) attacks, which is a stronger adaptive attack strategy. It can be easily integrated with existing attack algorithms to evade the SDMs by generating queries with fine-tuned direction and step size of perturbations utilizing the leaked decision information from the SDMs.   In this paper, we propose a novel approach, Query Provenance Analysis (QPA), for more robust and efficient SDMs. QPA encapsulates the historical relationships among queries as the sequence feature to capture the fundamental difference between benign and adversarial query sequences. To utilize the query provenance, we propose an efficient query provenance analysis algorithm with dynamic management. We evaluate QPA compared with two baselines, BlackLight and PIHA, on four widely used datasets with six query-based black-box attack algorithms. The results show that QPA outperforms the baselines in terms of defense effectiveness and efficiency on both non-adaptive and adaptive attacks. Specifically, QPA reduces the Attack Success Rate (ASR) of OARS to 4.08%, comparing to 77.63% and 87.72% for BlackLight and PIHA, respectively. Moreover, QPA also achieves 7.67x and 2.25x higher throughput than BlackLight and PIHA.



## **8. Disrupting Diffusion: Token-Level Attention Erasure Attack against Diffusion-based Customization**

cs.CV

Under review

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20584v1) [paper-pdf](http://arxiv.org/pdf/2405.20584v1)

**Authors**: Yisu Liu, Jinyang An, Wanqian Zhang, Dayan Wu, Jingzi Gu, Zheng Lin, Weiping Wang

**Abstract**: With the development of diffusion-based customization methods like DreamBooth, individuals now have access to train the models that can generate their personalized images. Despite the convenience, malicious users have misused these techniques to create fake images, thereby triggering a privacy security crisis. In light of this, proactive adversarial attacks are proposed to protect users against customization. The adversarial examples are trained to distort the customization model's outputs and thus block the misuse. In this paper, we propose DisDiff (Disrupting Diffusion), a novel adversarial attack method to disrupt the diffusion model outputs. We first delve into the intrinsic image-text relationships, well-known as cross-attention, and empirically find that the subject-identifier token plays an important role in guiding image generation. Thus, we propose the Cross-Attention Erasure module to explicitly "erase" the indicated attention maps and disrupt the text guidance. Besides,we analyze the influence of the sampling process of the diffusion model on Projected Gradient Descent (PGD) attack and introduce a novel Merit Sampling Scheduler to adaptively modulate the perturbation updating amplitude in a step-aware manner. Our DisDiff outperforms the state-of-the-art methods by 12.75% of FDFR scores and 7.25% of ISM scores across two facial benchmarks and two commonly used prompts on average.



## **9. Robustifying Safety-Aligned Large Language Models through Clean Data Curation**

cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.19358v2) [paper-pdf](http://arxiv.org/pdf/2405.19358v2)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are vulnerable when trained on datasets containing harmful content, which leads to potential jailbreaking attacks in two scenarios: the integration of harmful texts within crowdsourced data used for pre-training and direct tampering with LLMs through fine-tuning. In both scenarios, adversaries can compromise the safety alignment of LLMs, exacerbating malfunctions. Motivated by the need to mitigate these adversarial influences, our research aims to enhance safety alignment by either neutralizing the impact of malicious texts in pre-training datasets or increasing the difficulty of jailbreaking during downstream fine-tuning. In this paper, we propose a data curation framework designed to counter adversarial impacts in both scenarios. Our method operates under the assumption that we have no prior knowledge of attack details, focusing solely on curating clean texts. We introduce an iterative process aimed at revising texts to reduce their perplexity as perceived by LLMs, while simultaneously preserving their text quality. By pre-training or fine-tuning LLMs with curated clean texts, we observe a notable improvement in LLM robustness regarding safety alignment against harmful queries. For instance, when pre-training LLMs using a crowdsourced dataset containing 5\% harmful instances, adding an equivalent amount of curated texts significantly mitigates the likelihood of providing harmful responses in LLMs and reduces the attack success rate by 71\%. Our study represents a significant step towards mitigating the risks associated with training-based jailbreaking and fortifying the secure utilization of LLMs.



## **10. Extreme Miscalibration and the Illusion of Adversarial Robustness**

cs.CL

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2402.17509v2) [paper-pdf](http://arxiv.org/pdf/2402.17509v2)

**Authors**: Vyas Raina, Samson Tan, Volkan Cevher, Aditya Rawal, Sheng Zha, George Karypis

**Abstract**: Deep learning-based Natural Language Processing (NLP) models are vulnerable to adversarial attacks, where small perturbations can cause a model to misclassify. Adversarial Training (AT) is often used to increase model robustness. However, we have discovered an intriguing phenomenon: deliberately or accidentally miscalibrating models masks gradients in a way that interferes with adversarial attack search methods, giving rise to an apparent increase in robustness. We show that this observed gain in robustness is an illusion of robustness (IOR), and demonstrate how an adversary can perform various forms of test-time temperature calibration to nullify the aforementioned interference and allow the adversarial attack to find adversarial examples. Hence, we urge the NLP community to incorporate test-time temperature scaling into their robustness evaluations to ensure that any observed gains are genuine. Finally, we show how the temperature can be scaled during \textit{training} to improve genuine robustness.



## **11. Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers**

cs.RO

RSS 2024

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.12424v2) [paper-pdf](http://arxiv.org/pdf/2405.12424v2)

**Authors**: Fan Shi, Chong Zhang, Takahiro Miki, Joonho Lee, Marco Hutter, Stelian Coros

**Abstract**: Legged locomotion has recently achieved remarkable success with the progress of machine learning techniques, especially deep reinforcement learning (RL). Controllers employing neural networks have demonstrated empirical and qualitative robustness against real-world uncertainties, including sensor noise and external perturbations. However, formally investigating the vulnerabilities of these locomotion controllers remains a challenge. This difficulty arises from the requirement to pinpoint vulnerabilities across a long-tailed distribution within a high-dimensional, temporally sequential space. As a first step towards quantitative verification, we propose a computational method that leverages sequential adversarial attacks to identify weaknesses in learned locomotion controllers. Our research demonstrates that, even state-of-the-art robust controllers can fail significantly under well-designed, low-magnitude adversarial sequence. Through experiments in simulation and on the real robot, we validate our approach's effectiveness, and we illustrate how the results it generates can be used to robustify the original policy and offer valuable insights into the safety of these black-box policies. Project page: https://fanshi14.github.io/me/rss24.html



## **12. SleeperNets: Universal Backdoor Poisoning Attacks Against Reinforcement Learning Agents**

cs.LG

23 pages, 14 figures, NeurIPS

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20539v1) [paper-pdf](http://arxiv.org/pdf/2405.20539v1)

**Authors**: Ethan Rathbun, Christopher Amato, Alina Oprea

**Abstract**: Reinforcement learning (RL) is an actively growing field that is seeing increased usage in real-world, safety-critical applications -- making it paramount to ensure the robustness of RL algorithms against adversarial attacks. In this work we explore a particularly stealthy form of training-time attacks against RL -- backdoor poisoning. Here the adversary intercepts the training of an RL agent with the goal of reliably inducing a particular action when the agent observes a pre-determined trigger at inference time. We uncover theoretical limitations of prior work by proving their inability to generalize across domains and MDPs. Motivated by this, we formulate a novel poisoning attack framework which interlinks the adversary's objectives with those of finding an optimal policy -- guaranteeing attack success in the limit. Using insights from our theoretical analysis we develop ``SleeperNets'' as a universal backdoor attack which exploits a newly proposed threat model and leverages dynamic reward poisoning techniques. We evaluate our attack in 6 environments spanning multiple domains and demonstrate significant improvements in attack success over existing methods, while preserving benign episodic return.



## **13. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20485v1) [paper-pdf](http://arxiv.org/pdf/2405.20485v1)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs) in chatbot applications, enabling developers to adapt and personalize the LLM output without expensive training or fine-tuning. RAG systems use an external knowledge database to retrieve the most relevant documents for a given query, providing this context to the LLM generator. While RAG achieves impressive utility in many applications, its adoption to enable personalized generative models introduces new security risks. In this work, we propose new attack surfaces for an adversary to compromise a victim's RAG system, by injecting a single malicious document in its knowledge database. We design Phantom, general two-step attack framework against RAG augmented LLMs. The first step involves crafting a poisoned document designed to be retrieved by the RAG system within the top-k results only when an adversarial trigger, a specific sequence of words acting as backdoor, is present in the victim's queries. In the second step, a specially crafted adversarial string within the poisoned document triggers various adversarial attacks in the LLM generator, including denial of service, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama.



## **14. Tight Characterizations for Preprocessing against Cryptographic Salting**

cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20281v1) [paper-pdf](http://arxiv.org/pdf/2405.20281v1)

**Authors**: Fangqi Dong, Qipeng Liu, Kewen Wu

**Abstract**: Cryptography often considers the strongest yet plausible attacks in the real world. Preprocessing (a.k.a. non-uniform attack) plays an important role in both theory and practice: an efficient online attacker can take advantage of advice prepared by a time-consuming preprocessing stage.   Salting is a heuristic strategy to counter preprocessing attacks by feeding a small amount of randomness to the cryptographic primitive. We present general and tight characterizations of preprocessing against cryptographic salting, with upper bounds matching the advantages of the most intuitive attack. Our result quantitatively strengthens the previous work by Coretti, Dodis, Guo, and Steinberger (EUROCRYPT'18). Our proof exploits a novel connection between the non-uniform security of salted games and direct product theorems for memoryless algorithms.   For quantum adversaries, we give similar characterizations for property finding games, resolving an open problem of the quantum non-uniform security of salted collision resistant hash by Chung, Guo, Liu, and Qian (FOCS'20). Our proof extends the compressed oracle framework of Zhandry (CRYPTO'19) to prove quantum strong direct product theorems for property finding games in the average-case hardness.



## **15. AI-based Identification of Most Critical Cyberattacks in Industrial Systems**

eess.SY

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2306.04821v2) [paper-pdf](http://arxiv.org/pdf/2306.04821v2)

**Authors**: Bruno Paes Leao, Jagannadh Vempati, Siddharth Bhela, Tobias Ahlgrim, Daniel Arnold

**Abstract**: Modern industrial systems face a growing threat from sophisticated cyberattacks that can cause significant operational disruptions. This work presents a novel methodology for identification of the most critical cyberattacks that may disrupt the operation of such a system. Application of the proposed framework can enable the design and development of advanced cybersecurity solutions for a wide range of industrial applications. Attacks are assessed taking into direct consideration how they impact the system operation as measured by a defined Key Performance Indicator (KPI). A simulation model (SM), of the industrial process is employed for calculation of the KPI based on operating conditions. Such SM is augmented with a layer of information describing the communication network topology, connected devices, and potential actions an adversary can take based on each device or network link. Each possible action is associated with an abstract measure of effort, which is interpreted as a cost. It is assumed that the adversary has a corresponding budget that constrains the selection of the sequence of actions defining the progression of the attack. A dynamical system comprising a set of states associated with the cyberattack (cyber-states) and transition logic for updating their values is also proposed. The resulting augmented simulation model (ASM) is then employed in an artificial intelligence-based sequential decision-making optimization to yield the most critical cyberattack scenarios as measured by their impact on the defined KPI. The methodology is successfully tested based on an electrical power distribution system use case.



## **16. Lasso-based state estimation for cyber-physical systems under sensor attacks**

math.OC

\textcopyright 2024 the authors. This work has been accepted to IFAC  for publication under a Creative Commons Licence CC-BY-NC-ND

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20209v1) [paper-pdf](http://arxiv.org/pdf/2405.20209v1)

**Authors**: Vito Cerone, Sophie M. Fosson, Diego Regruto, Francesco Ripa

**Abstract**: The development of algorithms for secure state estimation in vulnerable cyber-physical systems has been gaining attention in the last years. A consolidated assumption is that an adversary can tamper a relatively small number of sensors. In the literature, block-sparsity methods exploit this prior information to recover the attack locations and the state of the system.   In this paper, we propose an alternative, Lasso-based approach and we analyse its effectiveness. In particular, we theoretically derive conditions that guarantee successful attack/state recovery, independently of established time sparsity patterns. Furthermore, we develop a sparse state observer, by starting from the iterative soft thresholding algorithm for Lasso, to perform online estimation.   Through several numerical experiments, we compare the proposed methods to the state-of-the-art algorithms.



## **17. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20090v1) [paper-pdf](http://arxiv.org/pdf/2405.20090v1)

**Authors**: Hao Cheng, Erjia Xiao, Jiahang Cao, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.



## **18. DiffPhysBA: Diffusion-based Physical Backdoor Attack against Person Re-Identification in Real-World**

cs.CV

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19990v1) [paper-pdf](http://arxiv.org/pdf/2405.19990v1)

**Authors**: Wenli Sun, Xinyang Jiang, Dongsheng Li, Cairong Zhao

**Abstract**: Person Re-Identification (ReID) systems pose a significant security risk from backdoor attacks, allowing adversaries to evade tracking or impersonate others. Beyond recognizing this issue, we investigate how backdoor attacks can be deployed in real-world scenarios, where a ReID model is typically trained on data collected in the digital domain and then deployed in a physical environment. This attack scenario requires an attack flow that embeds backdoor triggers in the digital domain realistically enough to also activate the buried backdoor in person ReID models in the physical domain. This paper realizes this attack flow by leveraging a diffusion model to generate realistic accessories on pedestrian images (e.g., bags, hats, etc.) as backdoor triggers. However, the noticeable domain gap between the triggers generated by the off-the-shelf diffusion model and their physical counterparts results in a low attack success rate. Therefore, we introduce a novel diffusion-based physical backdoor attack (DiffPhysBA) method that adopts a training-free similarity-guided sampling process to enhance the resemblance between generated and physical triggers. Consequently, DiffPhysBA can generate realistic attributes as semantic-level triggers in the digital domain and provides higher physical ASR compared to the direct paste method by 25.6% on the real-world test set. Through evaluations on newly proposed real-world and synthetic ReID test sets, DiffPhysBA demonstrates an impressive success rate exceeding 90% in both the digital and physical domains. Notably, it excels in digital stealth metrics and can effectively evade state-of-the-art defense methods.



## **19. HOLMES: to Detect Adversarial Examples with Multiple Detectors**

cs.AI

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19956v1) [paper-pdf](http://arxiv.org/pdf/2405.19956v1)

**Authors**: Jing Wen

**Abstract**: Deep neural networks (DNNs) can easily be cheated by some imperceptible but purposeful noise added to images, and erroneously classify them. Previous defensive work mostly focused on retraining the models or detecting the noise, but has either shown limited success rates or been attacked by new adversarial examples. Instead of focusing on adversarial images or the interior of DNN models, we observed that adversarial examples generated by different algorithms can be identified based on the output of DNNs (logits). Logit can serve as an exterior feature to train detectors. Then, we propose HOLMES (Hierarchically Organized Light-weight Multiple dEtector System) to reinforce DNNs by detecting potential adversarial examples to minimize the threats they may bring in practical. HOLMES is able to distinguish \textit{unseen} adversarial examples from multiple attacks with high accuracy and low false positive rates than single detector systems even in an adaptive model. To ensure the diversity and randomness of detectors in HOLMES, we use two methods: training dedicated detectors for each label and training detectors with top-k logits. Our effective and inexpensive strategies neither modify original DNN models nor require its internal parameters. HOLMES is not only compatible with all kinds of learning models (even only with external APIs), but also complementary to other defenses to achieve higher detection rates (may also fully protect the system against various adversarial examples).



## **20. BAN: Detecting Backdoors Activated by Adversarial Neuron Noise**

cs.LG

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19928v1) [paper-pdf](http://arxiv.org/pdf/2405.19928v1)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Shujian Yu, Stjepan Picek

**Abstract**: Backdoor attacks on deep learning represent a recent threat that has gained significant attention in the research community. Backdoor defenses are mainly based on backdoor inversion, which has been shown to be generic, model-agnostic, and applicable to practical threat scenarios. State-of-the-art backdoor inversion recovers a mask in the feature space to locate prominent backdoor features, where benign and backdoor features can be disentangled. However, it suffers from high computational overhead, and we also find that it overly relies on prominent backdoor features that are highly distinguishable from benign features. To tackle these shortcomings, this paper improves backdoor feature inversion for backdoor detection by incorporating extra neuron activation information. In particular, we adversarially increase the loss of backdoored models with respect to weights to activate the backdoor effect, based on which we can easily differentiate backdoored and clean models. Experimental results demonstrate our defense, BAN, is 1.37$\times$ (on CIFAR-10) and 5.11$\times$ (on ImageNet200) more efficient with 9.99% higher detect success rate than the state-of-the-art defense BTI-DBF. Our code and trained models are publicly available.\url{https://anonymous.4open.science/r/ban-4B32}



## **21. Robust Kernel Hypothesis Testing under Data Corruption**

stat.ML

26 pages, 2 figures, 2 algorithms

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19912v1) [paper-pdf](http://arxiv.org/pdf/2405.19912v1)

**Authors**: Antonin Schrab, Ilmun Kim

**Abstract**: We propose two general methods for constructing robust permutation tests under data corruption. The proposed tests effectively control the non-asymptotic type I error under data corruption, and we prove their consistency in power under minimal conditions. This contributes to the practical deployment of hypothesis tests for real-world applications with potential adversarial attacks. One of our methods inherently ensures differential privacy, further broadening its applicability to private data analysis. For the two-sample and independence settings, we show that our kernel robust tests are minimax optimal, in the sense that they are guaranteed to be non-asymptotically powerful against alternatives uniformly separated from the null in the kernel MMD and HSIC metrics at some optimal rate (tight with matching lower bound). Finally, we provide publicly available implementations and empirically illustrate the practicality of our proposed tests.



## **22. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

cs.MM

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19802v1) [paper-pdf](http://arxiv.org/pdf/2405.19802v1)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.



## **23. SimAC: A Simple Anti-Customization Method for Protecting Face Privacy against Text-to-Image Synthesis of Diffusion Models**

cs.CV

Accepted by CVPR2024

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2312.07865v3) [paper-pdf](http://arxiv.org/pdf/2312.07865v3)

**Authors**: Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, Qidong Huang

**Abstract**: Despite the success of diffusion-based customization methods on visual content creation, increasing concerns have been raised about such techniques from both privacy and political perspectives. To tackle this issue, several anti-customization methods have been proposed in very recent months, predominantly grounded in adversarial attacks. Unfortunately, most of these methods adopt straightforward designs, such as end-to-end optimization with a focus on adversarially maximizing the original training loss, thereby neglecting nuanced internal properties intrinsic to the diffusion model, and even leading to ineffective optimization in some diffusion time steps.In this paper, we strive to bridge this gap by undertaking a comprehensive exploration of these inherent properties, to boost the performance of current anti-customization approaches. Two aspects of properties are investigated: 1) We examine the relationship between time step selection and the model's perception in the frequency domain of images and find that lower time steps can give much more contributions to adversarial noises. This inspires us to propose an adaptive greedy search for optimal time steps that seamlessly integrates with existing anti-customization methods. 2) We scrutinize the roles of features at different layers during denoising and devise a sophisticated feature-based optimization framework for anti-customization.Experiments on facial benchmarks demonstrate that our approach significantly increases identity disruption, thereby protecting user privacy and copyright. Our code is available at: https://github.com/somuchtome/SimAC.



## **24. Enhancing Adversarial Robustness in SNNs with Sparse Gradients**

cs.NE

accepted by ICML 2024

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20355v1) [paper-pdf](http://arxiv.org/pdf/2405.20355v1)

**Authors**: Yujia Liu, Tong Bu, Jianhao Ding, Zecheng Hao, Tiejun Huang, Zhaofei Yu

**Abstract**: Spiking Neural Networks (SNNs) have attracted great attention for their energy-efficient operations and biologically inspired structures, offering potential advantages over Artificial Neural Networks (ANNs) in terms of energy efficiency and interpretability. Nonetheless, similar to ANNs, the robustness of SNNs remains a challenge, especially when facing adversarial attacks. Existing techniques, whether adapted from ANNs or specifically designed for SNNs, exhibit limitations in training SNNs or defending against strong attacks. In this paper, we propose a novel approach to enhance the robustness of SNNs through gradient sparsity regularization. We observe that SNNs exhibit greater resilience to random perturbations compared to adversarial perturbations, even at larger scales. Motivated by this, we aim to narrow the gap between SNNs under adversarial and random perturbations, thereby improving their overall robustness. To achieve this, we theoretically prove that this performance gap is upper bounded by the gradient sparsity of the probability associated with the true label concerning the input image, laying the groundwork for a practical strategy to train robust SNNs by regularizing the gradient sparsity. We validate the effectiveness of our approach through extensive experiments on both image-based and event-based datasets. The results demonstrate notable improvements in the robustness of SNNs. Our work highlights the importance of gradient sparsity in SNNs and its role in enhancing robustness.



## **25. Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack**

cs.CV

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.16134v2) [paper-pdf](http://arxiv.org/pdf/2405.16134v2)

**Authors**: Mingli Zhu, Siyuan Liang, Baoyuan Wu

**Abstract**: Deep neural networks face persistent challenges in defending against backdoor attacks, leading to an ongoing battle between attacks and defenses. While existing backdoor defense strategies have shown promising performance on reducing attack success rates, can we confidently claim that the backdoor threat has truly been eliminated from the model? To address it, we re-investigate the characteristics of the backdoored models after defense (denoted as defense models). Surprisingly, we find that the original backdoors still exist in defense models derived from existing post-training defense strategies, and the backdoor existence is measured by a novel metric called backdoor existence coefficient. It implies that the backdoors just lie dormant rather than being eliminated. To further verify this finding, we empirically show that these dormant backdoors can be easily re-activated during inference, by manipulating the original trigger with well-designed tiny perturbation using universal adversarial attack. More practically, we extend our backdoor reactivation to black-box scenario, where the defense model can only be queried by the adversary during inference, and develop two effective methods, i.e., query-based and transfer-based backdoor re-activation attacks. The effectiveness of the proposed methods are verified on both image classification and multimodal contrastive learning (i.e., CLIP) tasks. In conclusion, this work uncovers a critical vulnerability that has never been explored in existing defense strategies, emphasizing the urgency of designing more robust and advanced backdoor defense mechanisms in the future.



## **26. AutoBreach: Universal and Adaptive Jailbreaking with Efficient Wordplay-Guided Optimization**

cs.CV

Under review

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19668v1) [paper-pdf](http://arxiv.org/pdf/2405.19668v1)

**Authors**: Jiawei Chen, Xiao Yang, Zhengwei Fang, Yu Tian, Yinpeng Dong, Zhaoxia Yin, Hang Su

**Abstract**: Despite the widespread application of large language models (LLMs) across various tasks, recent studies indicate that they are susceptible to jailbreak attacks, which can render their defense mechanisms ineffective. However, previous jailbreak research has frequently been constrained by limited universality, suboptimal efficiency, and a reliance on manual crafting. In response, we rethink the approach to jailbreaking LLMs and formally define three essential properties from the attacker' s perspective, which contributes to guiding the design of jailbreak methods. We further introduce AutoBreach, a novel method for jailbreaking LLMs that requires only black-box access. Inspired by the versatility of wordplay, AutoBreach employs a wordplay-guided mapping rule sampling strategy to generate a variety of universal mapping rules for creating adversarial prompts. This generation process leverages LLMs' automatic summarization and reasoning capabilities, thus alleviating the manual burden. To boost jailbreak success rates, we further suggest sentence compression and chain-of-thought-based mapping rules to correct errors and wordplay misinterpretations in target LLMs. Additionally, we propose a two-stage mapping rule optimization strategy that initially optimizes mapping rules before querying target LLMs to enhance the efficiency of AutoBreach. AutoBreach can efficiently identify security vulnerabilities across various LLMs, including three proprietary models: Claude-3, GPT-3.5, GPT-4 Turbo, and two LLMs' web platforms: Bingchat, GPT-4 Web, achieving an average success rate of over 80% with fewer than 10 queries



## **27. Evaluating the Effectiveness and Robustness of Visual Similarity-based Phishing Detection Models**

cs.CR

12 pages

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19598v1) [paper-pdf](http://arxiv.org/pdf/2405.19598v1)

**Authors**: Fujiao Ji, Kiho Lee, Hyungjoon Koo, Wenhao You, Euijin Choo, Hyoungshick Kim, Doowon Kim

**Abstract**: Phishing attacks pose a significant threat to Internet users, with cybercriminals elaborately replicating the visual appearance of legitimate websites to deceive victims. Visual similarity-based detection systems have emerged as an effective countermeasure, but their effectiveness and robustness in real-world scenarios have been unexplored. In this paper, we comprehensively scrutinize and evaluate state-of-the-art visual similarity-based anti-phishing models using a large-scale dataset of 450K real-world phishing websites. Our analysis reveals that while certain models maintain high accuracy, others exhibit notably lower performance than results on curated datasets, highlighting the importance of real-world evaluation. In addition, we observe the real-world tactic of manipulating visual components that phishing attackers employ to circumvent the detection systems. To assess the resilience of existing models against adversarial attacks and robustness, we apply visible and perturbation-based manipulations to website logos, which adversaries typically target. We then evaluate the models' robustness in handling these adversarial samples. Our findings reveal vulnerabilities in several models, emphasizing the need for more robust visual similarity techniques capable of withstanding sophisticated evasion attempts. We provide actionable insights for enhancing the security of phishing defense systems, encouraging proactive actions. To the best of our knowledge, this work represents the first large-scale, systematic evaluation of visual similarity-based models for phishing detection in real-world settings, necessitating the development of more effective and robust defenses.



## **28. AI Risk Management Should Incorporate Both Safety and Security**

cs.CR

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19524v1) [paper-pdf](http://arxiv.org/pdf/2405.19524v1)

**Authors**: Xiangyu Qi, Yangsibo Huang, Yi Zeng, Edoardo Debenedetti, Jonas Geiping, Luxi He, Kaixuan Huang, Udari Madhushani, Vikash Sehwag, Weijia Shi, Boyi Wei, Tinghao Xie, Danqi Chen, Pin-Yu Chen, Jeffrey Ding, Ruoxi Jia, Jiaqi Ma, Arvind Narayanan, Weijie J Su, Mengdi Wang, Chaowei Xiao, Bo Li, Dawn Song, Peter Henderson, Prateek Mittal

**Abstract**: The exposure of security vulnerabilities in safety-aligned language models, e.g., susceptibility to adversarial attacks, has shed light on the intricate interplay between AI safety and AI security. Although the two disciplines now come together under the overarching goal of AI risk management, they have historically evolved separately, giving rise to differing perspectives. Therefore, in this paper, we advocate that stakeholders in AI risk management should be aware of the nuances, synergies, and interplay between safety and security, and unambiguously take into account the perspectives of both disciplines in order to devise mostly effective and holistic risk mitigation approaches. Unfortunately, this vision is often obfuscated, as the definitions of the basic concepts of "safety" and "security" themselves are often inconsistent and lack consensus across communities. With AI risk management being increasingly cross-disciplinary, this issue is particularly salient. In light of this conceptual challenge, we introduce a unified reference framework to clarify the differences and interplay between AI safety and AI security, aiming to facilitate a shared understanding and effective collaboration across communities.



## **29. IOI: Invisible One-Iteration Adversarial Attack on No-Reference Image- and Video-Quality Metrics**

eess.IV

Accepted to ICML 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2403.05955v2) [paper-pdf](http://arxiv.org/pdf/2403.05955v2)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: No-reference image- and video-quality metrics are widely used in video processing benchmarks. The robustness of learning-based metrics under video attacks has not been widely studied. In addition to having success, attacks that can be employed in video processing benchmarks must be fast and imperceptible. This paper introduces an Invisible One-Iteration (IOI) adversarial attack on no reference image and video quality metrics. We compared our method alongside eight prior approaches using image and video datasets via objective and subjective tests. Our method exhibited superior visual quality across various attacked metric architectures while maintaining comparable attack success and speed. We made the code available on GitHub: https://github.com/katiashh/ioi-attack.



## **30. Diffusion Policy Attacker: Crafting Adversarial Attacks for Diffusion-based Policies**

cs.CV

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19424v1) [paper-pdf](http://arxiv.org/pdf/2405.19424v1)

**Authors**: Yipu Chen, Haotian Xue, Yongxin Chen

**Abstract**: Diffusion models (DMs) have emerged as a promising approach for behavior cloning (BC). Diffusion policies (DP) based on DMs have elevated BC performance to new heights, demonstrating robust efficacy across diverse tasks, coupled with their inherent flexibility and ease of implementation. Despite the increasing adoption of DP as a foundation for policy generation, the critical issue of safety remains largely unexplored. While previous attempts have targeted deep policy networks, DP used diffusion models as the policy network, making it ineffective to be attacked using previous methods because of its chained structure and randomness injected. In this paper, we undertake a comprehensive examination of DP safety concerns by introducing adversarial scenarios, encompassing offline and online attacks, and global and patch-based attacks. We propose DP-Attacker, a suite of algorithms that can craft effective adversarial attacks across all aforementioned scenarios. We conduct attacks on pre-trained diffusion policies across various manipulation tasks. Through extensive experiments, we demonstrate that DP-Attacker has the capability to significantly decrease the success rate of DP for all scenarios. Particularly in offline scenarios, DP-Attacker can generate highly transferable perturbations applicable to all frames. Furthermore, we illustrate the creation of adversarial physical patches that, when applied to the environment, effectively deceive the model. Video results are put in: https://sites.google.com/view/diffusion-policy-attacker.



## **31. ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning**

cs.CV

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19237v1) [paper-pdf](http://arxiv.org/pdf/2405.19237v1)

**Authors**: Ruchika Chavhan, Da Li, Timothy Hospedales

**Abstract**: While large-scale text-to-image diffusion models have demonstrated impressive image-generation capabilities, there are significant concerns about their potential misuse for generating unsafe content, violating copyright, and perpetuating societal biases. Recently, the text-to-image generation community has begun addressing these concerns by editing or unlearning undesired concepts from pre-trained models. However, these methods often involve data-intensive and inefficient fine-tuning or utilize various forms of token remapping, rendering them susceptible to adversarial jailbreaks. In this paper, we present a simple and effective training-free approach, ConceptPrune, wherein we first identify critical regions within pre-trained models responsible for generating undesirable concepts, thereby facilitating straightforward concept unlearning via weight pruning. Experiments across a range of concepts including artistic styles, nudity, object erasure, and gender debiasing demonstrate that target concepts can be efficiently erased by pruning a tiny fraction, approximately 0.12% of total weights, enabling multi-concept erasure and robustness against various white-box and black-box adversarial attacks.



## **32. Gone but Not Forgotten: Improved Benchmarks for Machine Unlearning**

cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19211v1) [paper-pdf](http://arxiv.org/pdf/2405.19211v1)

**Authors**: Keltin Grimes, Collin Abidi, Cole Frank, Shannon Gallagher

**Abstract**: Machine learning models are vulnerable to adversarial attacks, including attacks that leak information about the model's training data. There has recently been an increase in interest about how to best address privacy concerns, especially in the presence of data-removal requests. Machine unlearning algorithms aim to efficiently update trained models to comply with data deletion requests while maintaining performance and without having to resort to retraining the model from scratch, a costly endeavor. Several algorithms in the machine unlearning literature demonstrate some level of privacy gains, but they are often evaluated only on rudimentary membership inference attacks, which do not represent realistic threats. In this paper we describe and propose alternative evaluation methods for three key shortcomings in the current evaluation of unlearning algorithms. We show the utility of our alternative evaluations via a series of experiments of state-of-the-art unlearning algorithms on different computer vision datasets, presenting a more detailed picture of the state of the field.



## **33. Model Agnostic Defense against Adversarial Patch Attacks on Object Detection in Unmanned Aerial Vehicles**

cs.CV

submitted to IROS 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19179v1) [paper-pdf](http://arxiv.org/pdf/2405.19179v1)

**Authors**: Saurabh Pathak, Samridha Shrestha, Abdelrahman AlMahmoud

**Abstract**: Object detection forms a key component in Unmanned Aerial Vehicles (UAVs) for completing high-level tasks that depend on the awareness of objects on the ground from an aerial perspective. In that scenario, adversarial patch attacks on an onboard object detector can severely impair the performance of upstream tasks. This paper proposes a novel model-agnostic defense mechanism against the threat of adversarial patch attacks in the context of UAV-based object detection. We formulate adversarial patch defense as an occlusion removal task. The proposed defense method can neutralize adversarial patches located on objects of interest, without exposure to adversarial patches during training. Our lightweight single-stage defense approach allows us to maintain a model-agnostic nature, that once deployed does not require to be updated in response to changes in the object detection pipeline. The evaluations in digital and physical domains show the feasibility of our method for deployment in UAV object detection pipelines, by significantly decreasing the Attack Success Ratio without incurring significant processing costs. As a result, the proposed defense solution can improve the reliability of object detection for UAVs.



## **34. Introducing Adaptive Continuous Adversarial Training (ACAT) to Enhance ML Robustness**

cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2403.10461v2) [paper-pdf](http://arxiv.org/pdf/2403.10461v2)

**Authors**: Mohamed elShehaby, Aditya Kotha, Ashraf Matrawy

**Abstract**: Adversarial training enhances the robustness of Machine Learning (ML) models against adversarial attacks. However, obtaining labeled training and adversarial training data in network/cybersecurity domains is challenging and costly. Therefore, this letter introduces Adaptive Continuous Adversarial Training (ACAT), a method that integrates adversarial training samples into the model during continuous learning sessions using real-world detected adversarial data. Experimental results with a SPAM detection dataset demonstrate that ACAT reduces the time required for adversarial sample detection compared to traditional processes. Moreover, the accuracy of the under-attack ML-based SPAM filter increased from 69% to over 88% after just three retraining sessions.



## **35. Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior**

cs.LG

ICML 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19098v1) [paper-pdf](http://arxiv.org/pdf/2405.19098v1)

**Authors**: Shuyu Cheng, Yibo Miao, Yinpeng Dong, Xiao Yang, Xiao-Shan Gao, Jun Zhu

**Abstract**: This paper studies the challenging black-box adversarial attack that aims to generate adversarial examples against a black-box model by only using output feedback of the model to input queries. Some previous methods improve the query efficiency by incorporating the gradient of a surrogate white-box model into query-based attacks due to the adversarial transferability. However, the localized gradient is not informative enough, making these methods still query-intensive. In this paper, we propose a Prior-guided Bayesian Optimization (P-BO) algorithm that leverages the surrogate model as a global function prior in black-box adversarial attacks. As the surrogate model contains rich prior information of the black-box one, P-BO models the attack objective with a Gaussian process whose mean function is initialized as the surrogate model's loss. Our theoretical analysis on the regret bound indicates that the performance of P-BO may be affected by a bad prior. Therefore, we further propose an adaptive integration strategy to automatically adjust a coefficient on the function prior by minimizing the regret bound. Extensive experiments on image classifiers and large vision-language models demonstrate the superiority of the proposed algorithm in reducing queries and improving attack success rates compared with the state-of-the-art black-box attacks. Code is available at https://github.com/yibo-miao/PBO-Attack.



## **36. New perspectives on the optimal placement of detectors for suicide bombers using metaheuristics**

cs.NE

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19060v1) [paper-pdf](http://arxiv.org/pdf/2405.19060v1)

**Authors**: Carlos Cotta, José E. Gallardo

**Abstract**: We consider an operational model of suicide bombing attacks -- an increasingly prevalent form of terrorism -- against specific targets, and the use of protective countermeasures based on the deployment of detectors over the area under threat. These detectors have to be carefully located in order to minimize the expected number of casualties or the economic damage suffered, resulting in a hard optimization problem for which different metaheuristics have been proposed. Rather than assuming random decisions by the attacker, the problem is approached by considering different models of the latter, whereby he takes informed decisions on which objective must be targeted and through which path it has to be reached based on knowledge on the importance or value of the objectives or on the defensive strategy of the defender (a scenario that can be regarded as an adversarial game). We consider four different algorithms, namely a greedy heuristic, a hill climber, tabu search and an evolutionary algorithm, and study their performance on a broad collection of problem instances trying to resemble different realistic settings such as a coastal area, a modern urban area, and the historic core of an old town. It is shown that the adversarial scenario is harder for all techniques, and that the evolutionary algorithm seems to adapt better to the complexity of the resulting search landscape.



## **37. Verifiably Robust Conformal Prediction**

cs.LO

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18942v1) [paper-pdf](http://arxiv.org/pdf/2405.18942v1)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces \emph{VRCP (Verifiably Robust Conformal Prediction)}, a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.



## **38. Proactive Load-Shaping Strategies with Privacy-Cost Trade-offs in Residential Households based on Deep Reinforcement Learning**

eess.SY

7 pages

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18888v1) [paper-pdf](http://arxiv.org/pdf/2405.18888v1)

**Authors**: Ruichang Zhang, Youcheng Sun, Mustafa A. Mustafa

**Abstract**: Smart meters play a crucial role in enhancing energy management and efficiency, but they raise significant privacy concerns by potentially revealing detailed user behaviors through energy consumption patterns. Recent scholarly efforts have focused on developing battery-aided load-shaping techniques to protect user privacy while balancing costs. This paper proposes a novel deep reinforcement learning-based load-shaping algorithm (PLS-DQN) designed to protect user privacy by proactively creating artificial load signatures that mislead potential attackers. We evaluate our proposed algorithm against a non-intrusive load monitoring (NILM) adversary. The results demonstrate that our approach not only effectively conceals real energy usage patterns but also outperforms state-of-the-art methods in enhancing user privacy while maintaining cost efficiency.



## **39. Enhancing Security and Privacy in Federated Learning using Update Digests and Voting-Based Defense**

cs.CR

14 pages

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18802v1) [paper-pdf](http://arxiv.org/pdf/2405.18802v1)

**Authors**: Wenjie Li, Kai Fan, Jingyuan Zhang, Hui Li, Wei Yang Bryan Lim, Qiang Yang

**Abstract**: Federated Learning (FL) is a promising privacy-preserving machine learning paradigm that allows data owners to collaboratively train models while keeping their data localized. Despite its potential, FL faces challenges related to the trustworthiness of both clients and servers, especially in the presence of curious or malicious adversaries. In this paper, we introduce a novel framework named \underline{\textbf{F}}ederated \underline{\textbf{L}}earning with \underline{\textbf{U}}pdate \underline{\textbf{D}}igest (FLUD), which addresses the critical issues of privacy preservation and resistance to Byzantine attacks within distributed learning environments. FLUD utilizes an innovative approach, the $\mathsf{LinfSample}$ method, allowing clients to compute the $l_{\infty}$ norm across sliding windows of updates as an update digest. This digest enables the server to calculate a shared distance matrix, significantly reducing the overhead associated with Secure Multi-Party Computation (SMPC) by three orders of magnitude while effectively distinguishing between benign and malicious updates. Additionally, FLUD integrates a privacy-preserving, voting-based defense mechanism that employs optimized SMPC protocols to minimize communication rounds. Our comprehensive experiments demonstrate FLUD's effectiveness in countering Byzantine adversaries while incurring low communication and runtime overhead. FLUD offers a scalable framework for secure and reliable FL in distributed environments, facilitating its application in scenarios requiring robust data management and security.



## **40. MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training**

cs.CR

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2311.00919v2) [paper-pdf](http://arxiv.org/pdf/2311.00919v2)

**Authors**: Jiacheng Li, Ninghui Li, Bruno Ribeiro

**Abstract**: In Member Inference (MI) attacks, the adversary try to determine whether an instance is used to train a machine learning (ML) model. MI attacks are a major privacy concern when using private data to train ML models. Most MI attacks in the literature take advantage of the fact that ML models are trained to fit the training data well, and thus have very low loss on training instances. Most defenses against MI attacks therefore try to make the model fit the training data less well. Doing so, however, generally results in lower accuracy. We observe that training instances have different degrees of vulnerability to MI attacks. Most instances will have low loss even when not included in training. For these instances, the model can fit them well without concerns of MI attacks. An effective defense only needs to (possibly implicitly) identify instances that are vulnerable to MI attacks and avoids overfitting them. A major challenge is how to achieve such an effect in an efficient training process. Leveraging two distinct recent advancements in representation learning: counterfactually-invariant representations and subspace learning methods, we introduce a novel Membership-Invariant Subspace Training (MIST) method to defend against MI attacks. MIST avoids overfitting the vulnerable instances without significant impact on other instances. We have conducted extensive experimental studies, comparing MIST with various other state-of-the-art (SOTA) MI defenses against several SOTA MI attacks. We find that MIST outperforms other defenses while resulting in minimal reduction in testing accuracy.



## **41. Leveraging Many-To-Many Relationships for Defending Against Visual-Language Adversarial Attacks**

cs.CV

Under review

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18770v1) [paper-pdf](http://arxiv.org/pdf/2405.18770v1)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos

**Abstract**: Recent studies have revealed that vision-language (VL) models are vulnerable to adversarial attacks for image-text retrieval (ITR). However, existing defense strategies for VL models primarily focus on zero-shot image classification, which do not consider the simultaneous manipulation of image and text, as well as the inherent many-to-many (N:N) nature of ITR, where a single image can be described in numerous ways, and vice versa. To this end, this paper studies defense strategies against adversarial attacks on VL models for ITR for the first time. Particularly, we focus on how to leverage the N:N relationship in ITR to enhance adversarial robustness. We found that, although adversarial training easily overfits to specific one-to-one (1:1) image-text pairs in the train data, diverse augmentation techniques to create one-to-many (1:N) / many-to-one (N:1) image-text pairs can significantly improve adversarial robustness in VL models. Additionally, we show that the alignment of the augmented image-text pairs is crucial for the effectiveness of the defense strategy, and that inappropriate augmentations can even degrade the model's performance. Based on these findings, we propose a novel defense strategy that leverages the N:N relationship in ITR, which effectively generates diverse yet highly-aligned N:N pairs using basic augmentations and generative model-based augmentations. This work provides a novel perspective on defending against adversarial attacks in VL tasks and opens up new research directions for future work.



## **42. Genshin: General Shield for Natural Language Processing with Large Language Models**

cs.CL

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18741v1) [paper-pdf](http://arxiv.org/pdf/2405.18741v1)

**Authors**: Xiao Peng, Tao Liu, Ying Wang

**Abstract**: Large language models (LLMs) like ChatGPT, Gemini, or LLaMA have been trending recently, demonstrating considerable advancement and generalizability power in countless domains. However, LLMs create an even bigger black box exacerbating opacity, with interpretability limited to few approaches. The uncertainty and opacity embedded in LLMs' nature restrict their application in high-stakes domains like financial fraud, phishing, etc. Current approaches mainly rely on traditional textual classification with posterior interpretable algorithms, suffering from attackers who may create versatile adversarial samples to break the system's defense, forcing users to make trade-offs between efficiency and robustness. To address this issue, we propose a novel cascading framework called Genshin (General Shield for Natural Language Processing with Large Language Models), utilizing LLMs as defensive one-time plug-ins. Unlike most applications of LLMs that try to transform text into something new or structural, Genshin uses LLMs to recover text to its original state. Genshin aims to combine the generalizability of the LLM, the discrimination of the median model, and the interpretability of the simple model. Our experiments on the task of sentimental analysis and spam detection have shown fatal flaws of the current median models and exhilarating results on LLMs' recovery ability, demonstrating that Genshin is both effective and efficient. In our ablation study, we unearth several intriguing observations. Utilizing the LLM defender, a tool derived from the 4th paradigm, we have reproduced BERT's 15% optimal mask rate results in the 3rd paradigm of NLP. Additionally, when employing the LLM as a potential adversarial tool, attackers are capable of executing effective attacks that are nearly semantically lossless.



## **43. Security--Throughput Tradeoff of Nakamoto Consensus under Bandwidth Constraints**

cs.CR

ACM Conference on Computer and Communications Security 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2303.09113v3) [paper-pdf](http://arxiv.org/pdf/2303.09113v3)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: For Nakamoto's longest-chain consensus protocol, whose proof-of-work (PoW) and proof-of-stake (PoS) variants power major blockchains such as Bitcoin and Cardano, we revisit the classic problem of the security-performance tradeoff: Given a network of nodes with limited capacities, against what fraction of adversary power is Nakamoto consensus (NC) secure for a given block production rate? State-of-the-art analyses of Nakamoto's protocol fail to answer this question because their bounded-delay model does not capture realistic constraints such as limited communication- and computation-resources. We develop a new analysis technique to prove a refined security-performance tradeoff for PoW Nakamoto consensus in a bounded-bandwidth model. In this model, we show that, in contrast to the classic bounded-delay model, Nakamoto's private attack is no longer the worst attack, and a new attack strategy we call the teasing strategy, that exploits the network congestion caused by limited bandwidth, is strictly worse. In PoS, equivocating blocks can exacerbate congestion, making the traditional PoS Nakamoto consensus protocol insecure except at very low block production rates. To counter such equivocation spamming, we present a variant of the PoS NC protocol we call Blanking NC (BlaNC), which achieves the same resilience as PoW NC.



## **44. Watermarking Counterfactual Explanations**

cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18671v1) [paper-pdf](http://arxiv.org/pdf/2405.18671v1)

**Authors**: Hangzhi Guo, Amulya Yadav

**Abstract**: The field of Explainable Artificial Intelligence (XAI) focuses on techniques for providing explanations to end-users about the decision-making processes that underlie modern-day machine learning (ML) models. Within the vast universe of XAI techniques, counterfactual (CF) explanations are often preferred by end-users as they help explain the predictions of ML models by providing an easy-to-understand & actionable recourse (or contrastive) case to individual end-users who are adversely impacted by predicted outcomes. However, recent studies have shown significant security concerns with using CF explanations in real-world applications; in particular, malicious adversaries can exploit CF explanations to perform query-efficient model extraction attacks on proprietary ML models. In this paper, we propose a model-agnostic watermarking framework (for adding watermarks to CF explanations) that can be leveraged to detect unauthorized model extraction attacks (which rely on the watermarked CF explanations). Our novel framework solves a bi-level optimization problem to embed an indistinguishable watermark into the generated CF explanation such that any future model extraction attacks that rely on these watermarked CF explanations can be detected using a null hypothesis significance testing (NHST) scheme, while ensuring that these embedded watermarks do not compromise the quality of the generated CF explanations. We evaluate this framework's performance across a diverse set of real-world datasets, CF explanation methods, and model extraction techniques, and show that our watermarking detection system can be used to accurately identify extracted ML models that are trained using the watermarked CF explanations. Our work paves the way for the secure adoption of CF explanations in real-world applications.



## **45. PureEBM: Universal Poison Purification via Mid-Run Dynamics of Energy-Based Models**

cs.LG

arXiv admin note: substantial text overlap with arXiv:2405.18627

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.19376v1) [paper-pdf](http://arxiv.org/pdf/2405.19376v1)

**Authors**: Omead Pooladzandi, Jeffrey Jiang, Sunay Bhat, Gregory Pottie

**Abstract**: Data poisoning attacks pose a significant threat to the integrity of machine learning models by leading to misclassification of target distribution test data by injecting adversarial examples during training. Existing state-of-the-art (SoTA) defense methods suffer from a variety of limitations, such as significantly reduced generalization performance, specificity to particular attack types and classifiers, and significant overhead during training, making them impractical or limited for real-world applications. In response to this challenge, we introduce a universal data purification method that defends naturally trained classifiers from malicious white-, gray-, and black-box image poisons by applying a universal stochastic preprocessing step $\Psi_{T}(x)$, realized by iterative Langevin sampling of a convergent Energy Based Model (EBM) initialized with an image $x.$ Mid-run dynamics of $\Psi_{T}(x)$ purify poison information with minimal impact on features important to the generalization of a classifier network. We show that the contrastive learning process of EBMs allows them to remain universal purifiers, even in the presence of poisoned EBM training data, and to achieve SoTA defense on leading triggered poison Narcissus and triggerless poisons Gradient Matching and Bullseye Polytope. This work is a subset of a larger framework introduced in PureGen with a more detailed focus on EBM purification and poison defense.



## **46. PureGen: Universal Data Purification for Train-Time Poison Defense via Generative Model Dynamics**

cs.LG

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18627v1) [paper-pdf](http://arxiv.org/pdf/2405.18627v1)

**Authors**: Sunay Bhat, Jeffrey Jiang, Omead Pooladzandi, Alexander Branch, Gregory Pottie

**Abstract**: Train-time data poisoning attacks threaten machine learning models by introducing adversarial examples during training, leading to misclassification. Current defense methods often reduce generalization performance, are attack-specific, and impose significant training overhead. To address this, we introduce a set of universal data purification methods using a stochastic transform, $\Psi(x)$, realized via iterative Langevin dynamics of Energy-Based Models (EBMs), Denoising Diffusion Probabilistic Models (DDPMs), or both. These approaches purify poisoned data with minimal impact on classifier generalization. Our specially trained EBMs and DDPMs provide state-of-the-art defense against various attacks (including Narcissus, Bullseye Polytope, Gradient Matching) on CIFAR-10, Tiny-ImageNet, and CINIC-10, without needing attack or classifier-specific information. We discuss performance trade-offs and show that our methods remain highly effective even with poisoned or distributionally shifted generative model training data.



## **47. Wavelet-Based Image Tokenizer for Vision Transformers**

cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18616v1) [paper-pdf](http://arxiv.org/pdf/2405.18616v1)

**Authors**: Zhenhai Zhu, Radu Soricut

**Abstract**: Non-overlapping patch-wise convolution is the default image tokenizer for all state-of-the-art vision Transformer (ViT) models. Even though many ViT variants have been proposed to improve its efficiency and accuracy, little research on improving the image tokenizer itself has been reported in the literature. In this paper, we propose a new image tokenizer based on wavelet transformation. We show that ViT models with the new tokenizer achieve both higher training throughput and better top-1 precision for the ImageNet validation set. We present a theoretical analysis on why the proposed tokenizer improves the training throughput without any change to ViT model architecture. Our analysis suggests that the new tokenizer can effectively handle high-resolution images and is naturally resistant to adversarial attack. Furthermore, the proposed image tokenizer offers a fresh perspective on important new research directions for ViT-based model design, such as image tokens on a non-uniform grid for image understanding.



## **48. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2402.17012v2) [paper-pdf](http://arxiv.org/pdf/2402.17012v2)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model which leverages recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Taken together, these results represent the strongest existing privacy attacks against both pretrained and fine-tuned LLMs for MIAs and training data extraction, which are of independent scientific interest and have important practical implications for LLM security, privacy, and copyright issues.



## **49. Unleashing the potential of prompt engineering: a comprehensive review**

cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2310.14735v3) [paper-pdf](http://arxiv.org/pdf/2310.14735v3)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review explores the transformative potential of prompt engineering within the realm of large language models (LLMs) and multimodal language models (MMLMs). The development of AI, from its inception in the 1950s to the emergence of neural networks and deep learning architectures, has culminated in sophisticated LLMs like GPT-4 and BERT, as well as MMLMs like DALL-E and CLIP. These models have revolutionized tasks in diverse fields such as workplace automation, healthcare, and education. Prompt engineering emerges as a crucial technique to maximize the utility and accuracy of these models. This paper delves into both foundational and advanced methodologies of prompt engineering, including techniques like Chain of Thought, Self-consistency, and Generated Knowledge, which significantly enhance model performance. Additionally, it examines the integration of multimodal data through innovative approaches such as Multi-modal Prompt Learning (MaPLe), Conditional Prompt Learning, and Context Optimization. Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review underscores the pivotal role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.



## **50. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

cs.AI

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18166v1) [paper-pdf](http://arxiv.org/pdf/2405.18166v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.



