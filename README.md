# Latest Adversarial Attack Papers
**update at 2024-03-07 10:25:33**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Improving Adversarial Attacks on Latent Diffusion Model**

cs.CV

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2310.04687v3) [paper-pdf](http://arxiv.org/pdf/2310.04687v3)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Adversarial attacks on Latent Diffusion Model (LDM), the state-of-the-art image generative model, have been adopted as effective protection against malicious finetuning of LDM on unauthorized images. We show that these attacks add an extra error to the score function of adversarial examples predicted by LDM. LDM finetuned on these adversarial examples learns to lower the error by a bias, from which the model is attacked and predicts the score function with biases.   Based on the dynamics, we propose to improve the adversarial attack on LDM by Attacking with Consistent score-function Errors (ACE). ACE unifies the pattern of the extra error added to the predicted score function. This induces the finetuned LDM to learn the same pattern as a bias in predicting the score function. We then introduce a well-crafted pattern to improve the attack. Our method outperforms state-of-the-art methods in adversarial attacks on LDM.



## **2. A Survey on Adversarial Contention Resolution**

cs.DC

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03876v1) [paper-pdf](http://arxiv.org/pdf/2403.03876v1)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.



## **3. Neural Exec: Learning (and Learning from) Execution Triggers for Prompt Injection Attacks**

cs.CR

v0.1

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03792v1) [paper-pdf](http://arxiv.org/pdf/2403.03792v1)

**Authors**: Dario Pasquini, Martin Strohmeier, Carmela Troncoso

**Abstract**: We introduce a new family of prompt injection attacks, termed Neural Exec. Unlike known attacks that rely on handcrafted strings (e.g., "Ignore previous instructions and..."), we show that it is possible to conceptualize the creation of execution triggers as a differentiable search problem and use learning-based methods to autonomously generate them.   Our results demonstrate that a motivated adversary can forge triggers that are not only drastically more effective than current handcrafted ones but also exhibit inherent flexibility in shape, properties, and functionality. In this direction, we show that an attacker can design and generate Neural Execs capable of persisting through multi-stage preprocessing pipelines, such as in the case of Retrieval-Augmented Generation (RAG)-based applications. More critically, our findings show that attackers can produce triggers that deviate markedly in form and shape from any known attack, sidestepping existing blacklist-based detection and sanitation approaches.



## **4. PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion**

cs.CL

LLM evaluation, Multi-turn, Multi-language, Multi-modal benchmark

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03788v1) [paper-pdf](http://arxiv.org/pdf/2403.03788v1)

**Authors**: Zekai Zhang, Yiduo Guo, Yaobo Liang, Dongyan Zhao, Nan Duan

**Abstract**: The growing dependence on Large Language Models (LLMs) for finishing user instructions necessitates a comprehensive understanding of their robustness to complex task completion in real-world situations. To address this critical need, we propose the PowerPoint Task Completion Robustness benchmark (PPTC-R) to measure LLMs' robustness to the user PPT task instruction and software version. Specifically, we construct adversarial user instructions by attacking user instructions at sentence, semantic, and multi-language levels. To assess the robustness of Language Models to software versions, we vary the number of provided APIs to simulate both the newest version and earlier version settings. Subsequently, we test 3 closed-source and 4 open-source LLMs using a benchmark that incorporates these robustness settings, aiming to evaluate how deviations impact LLMs' API calls for task completion. We find that GPT-4 exhibits the highest performance and strong robustness in our benchmark, particularly in the version update and the multilingual settings. However, we find that all LLMs lose their robustness when confronted with multiple challenges (e.g., multi-turn) simultaneously, leading to significant performance drops. We further analyze the robustness behavior and error reasons of LLMs in our benchmark, which provide valuable insights for researchers to understand the LLM's robustness in task completion and develop more robust LLMs and agents. We release the code and data at \url{https://github.com/ZekaiGalaxy/PPTCR}.



## **5. Verification of Neural Networks' Global Robustness**

cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.19322v2) [paper-pdf](http://arxiv.org/pdf/2402.19322v2)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstract**: Neural networks are successful in various applications but are also susceptible to adversarial attacks. To show the safety of network classifiers, many verifiers have been introduced to reason about the local robustness of a given input to a given perturbation. While successful, local robustness cannot generalize to unseen inputs. Several works analyze global robustness properties, however, neither can provide a precise guarantee about the cases where a network classifier does not change its classification. In this work, we propose a new global robustness property for classifiers aiming at finding the minimal globally robust bound, which naturally extends the popular local robustness property for classifiers. We introduce VHAGaR, an anytime verifier for computing this bound. VHAGaR relies on three main ideas: encoding the problem as a mixed-integer programming and pruning the search space by identifying dependencies stemming from the perturbation or the network's computation and generalizing adversarial attacks to unknown inputs. We evaluate VHAGaR on several datasets and classifiers and show that, given a three hour timeout, the average gap between the lower and upper bound on the minimal globally robust bound computed by VHAGaR is 1.9, while the gap of an existing global robustness verifier is 154.7. Moreover, VHAGaR is 130.6x faster than this verifier. Our results further indicate that leveraging dependencies and adversarial attacks makes VHAGaR 78.6x faster.



## **6. Simplified PCNet with Robustness**

cs.LG

10 pages, 3 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03676v1) [paper-pdf](http://arxiv.org/pdf/2403.03676v1)

**Authors**: Bingheng Li, Xuanting Xie, Haoxiang Lei, Ruiyi Fang, Zhao Kang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant attention for their success in learning the representation of homophilic or heterophilic graphs. However, they cannot generalize well to real-world graphs with different levels of homophily. In response, the Possion-Charlier Network (PCNet) \cite{li2024pc}, the previous work, allows graph representation to be learned from heterophily to homophily. Although PCNet alleviates the heterophily issue, there remain some challenges in further improving the efficacy and efficiency. In this paper, we simplify PCNet and enhance its robustness. We first extend the filter order to continuous values and reduce its parameters. Two variants with adaptive neighborhood sizes are implemented. Theoretical analysis shows our model's robustness to graph structure perturbations or adversarial attacks. We validate our approach through semi-supervised learning tasks on various datasets representing both homophilic and heterophilic graphs.



## **7. Adversarial Infrared Geometry: Using Geometry to Perform Adversarial Attack against Infrared Pedestrian Detectors**

cs.CV

arXiv admin note: text overlap with arXiv:2312.14217 by other authors

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03674v1) [paper-pdf](http://arxiv.org/pdf/2403.03674v1)

**Authors**: Kalibinuer Tiliwalidi

**Abstract**: Currently, infrared imaging technology enjoys widespread usage, with infrared object detection technology experiencing a surge in prominence. While previous studies have delved into physical attacks on infrared object detectors, the implementation of these techniques remains complex. For instance, some approaches entail the use of bulb boards or infrared QR suits as perturbations to execute attacks, which entail costly optimization and cumbersome deployment processes. Other methodologies involve the utilization of irregular aerogel as physical perturbations for infrared attacks, albeit at the expense of optimization expenses and perceptibility issues. In this study, we propose a novel infrared physical attack termed Adversarial Infrared Geometry (\textbf{AdvIG}), which facilitates efficient black-box query attacks by modeling diverse geometric shapes (lines, triangles, ellipses) and optimizing their physical parameters using Particle Swarm Optimization (PSO). Extensive experiments are conducted to evaluate the effectiveness, stealthiness, and robustness of AdvIG. In digital attack experiments, line, triangle, and ellipse patterns achieve attack success rates of 93.1\%, 86.8\%, and 100.0\%, respectively, with average query times of 71.7, 113.1, and 2.57, respectively, thereby confirming the efficiency of AdvIG. Physical attack experiments are conducted to assess the attack success rate of AdvIG at different distances. On average, the line, triangle, and ellipse achieve attack success rates of 61.1\%, 61.2\%, and 96.2\%, respectively. Further experiments are conducted to comprehensively analyze AdvIG, including ablation experiments, transfer attack experiments, and adversarial defense mechanisms. Given the superior performance of our method as a simple and efficient black-box adversarial attack in both digital and physical environments, we advocate for widespread attention to AdvIG.



## **8. Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning**

cs.CR

This article has been accepted to USENIX Security '24

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2401.02880v2) [paper-pdf](http://arxiv.org/pdf/2401.02880v2)

**Authors**: Zhifeng Jiang, Peng Ye, Shiqi He, Wei Wang, Ruichuan Chen, Bo Li

**Abstract**: In Federated Learning (FL), common privacy-enhancing techniques, such as secure aggregation and distributed differential privacy, rely on the critical assumption of an honest majority among participants to withstand various attacks. In practice, however, servers are not always trusted, and an adversarial server can strategically select compromised clients to create a dishonest majority, thereby undermining the system's security guarantees. In this paper, we present Lotto, an FL system that addresses this fundamental, yet underexplored issue by providing secure participant selection against an adversarial server. Lotto supports two selection algorithms: random and informed. To ensure random selection without a trusted server, Lotto enables each client to autonomously determine their participation using verifiable randomness. For informed selection, which is more vulnerable to manipulation, Lotto approximates the algorithm by employing random selection within a refined client pool. Our theoretical analysis shows that Lotto effectively aligns the proportion of server-selected compromised participants with the base rate of dishonest clients in the population. Large-scale experiments further reveal that Lotto achieves time-to-accuracy performance comparable to that of insecure selection methods, indicating a low computational overhead for secure selection.



## **9. Noise-BERT: A Unified Perturbation-Robust Framework with Noise Alignment Pre-training for Noisy Slot Filling Task**

cs.CL

Accepted by ICASSP 2024

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.14494v3) [paper-pdf](http://arxiv.org/pdf/2402.14494v3)

**Authors**: Jinxu Zhao, Guanting Dong, Yueyan Qiu, Tingfeng Hui, Xiaoshuai Song, Daichi Guo, Weiran Xu

**Abstract**: In a realistic dialogue system, the input information from users is often subject to various types of input perturbations, which affects the slot-filling task. Although rule-based data augmentation methods have achieved satisfactory results, they fail to exhibit the desired generalization when faced with unknown noise disturbances. In this study, we address the challenges posed by input perturbations in slot filling by proposing Noise-BERT, a unified Perturbation-Robust Framework with Noise Alignment Pre-training. Our framework incorporates two Noise Alignment Pre-training tasks: Slot Masked Prediction and Sentence Noisiness Discrimination, aiming to guide the pre-trained language model in capturing accurate slot information and noise distribution. During fine-tuning, we employ a contrastive learning loss to enhance the semantic representation of entities and labels. Additionally, we introduce an adversarial attack training strategy to improve the model's robustness. Experimental results demonstrate the superiority of our proposed approach over state-of-the-art models, and further analysis confirms its effectiveness and generalization ability.



## **10. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

cs.CL

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2312.14197v2) [paper-pdf](http://arxiv.org/pdf/2312.14197v2)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.



## **11. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs form Finished Cyber Threat Reports**

cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.03267v1) [paper-pdf](http://arxiv.org/pdf/2403.03267v1)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.



## **12. Attacks on Node Attributes in Graph Neural Networks**

cs.SI

Accepted to AAAI 2024 AICS workshop

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2402.12426v2) [paper-pdf](http://arxiv.org/pdf/2402.12426v2)

**Authors**: Ying Xu, Michael Lanier, Anindya Sarkar, Yevgeniy Vorobeychik

**Abstract**: Graphs are commonly used to model complex networks prevalent in modern social media and literacy applications. Our research investigates the vulnerability of these graphs through the application of feature based adversarial attacks, focusing on both decision time attacks and poisoning attacks. In contrast to state of the art models like Net Attack and Meta Attack, which target node attributes and graph structure, our study specifically targets node attributes. For our analysis, we utilized the text dataset Hellaswag and graph datasets Cora and CiteSeer, providing a diverse basis for evaluation. Our findings indicate that decision time attacks using Projected Gradient Descent (PGD) are more potent compared to poisoning attacks that employ Mean Node Embeddings and Graph Contrastive Learning strategies. This provides insights for graph data security, pinpointing where graph-based models are most vulnerable and thereby informing the development of stronger defense mechanisms against such attacks.



## **13. Mitigating Label Flipping Attacks in Malicious URL Detectors Using Ensemble Trees**

cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02995v1) [paper-pdf](http://arxiv.org/pdf/2403.02995v1)

**Authors**: Ehsan Nowroozi, Nada Jadalla, Samaneh Ghelichkhani, Alireza Jolfaei

**Abstract**: Malicious URLs provide adversarial opportunities across various industries, including transportation, healthcare, energy, and banking which could be detrimental to business operations. Consequently, the detection of these URLs is of crucial importance; however, current Machine Learning (ML) models are susceptible to backdoor attacks. These attacks involve manipulating a small percentage of training data labels, such as Label Flipping (LF), which changes benign labels to malicious ones and vice versa. This manipulation results in misclassification and leads to incorrect model behavior. Therefore, integrating defense mechanisms into the architecture of ML models becomes an imperative consideration to fortify against potential attacks.   The focus of this study is on backdoor attacks in the context of URL detection using ensemble trees. By illuminating the motivations behind such attacks, highlighting the roles of attackers, and emphasizing the critical importance of effective defense strategies, this paper contributes to the ongoing efforts to fortify ML models against adversarial threats within the ML domain in network security. We propose an innovative alarm system that detects the presence of poisoned labels and a defense mechanism designed to uncover the original class labels with the aim of mitigating backdoor attacks on ensemble tree classifiers. We conducted a case study using the Alexa and Phishing Site URL datasets and showed that LF attacks can be addressed using our proposed defense mechanism. Our experimental results prove that the LF attack achieved an Attack Success Rate (ASR) between 50-65% within 2-5%, and the innovative defense method successfully detected poisoned labels with an accuracy of up to 100%.



## **14. Federated Learning Under Attack: Exposing Vulnerabilities through Data Poisoning Attacks in Computer Networks**

cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02983v1) [paper-pdf](http://arxiv.org/pdf/2403.02983v1)

**Authors**: Ehsan Nowroozi, Imran Haider, Rahim Taheri, Mauro Conti

**Abstract**: Federated Learning (FL) is a machine learning (ML) approach that enables multiple decentralized devices or edge servers to collaboratively train a shared model without exchanging raw data. During the training and sharing of model updates between clients and servers, data and models are susceptible to different data-poisoning attacks.   In this study, our motivation is to explore the severity of data poisoning attacks in the computer network domain because they are easy to implement but difficult to detect. We considered two types of data-poisoning attacks, label flipping (LF) and feature poisoning (FP), and applied them with a novel approach. In LF, we randomly flipped the labels of benign data and trained the model on the manipulated data. For FP, we randomly manipulated the highly contributing features determined using the Random Forest algorithm. The datasets used in this experiment were CIC and UNSW related to computer networks. We generated adversarial samples using the two attacks mentioned above, which were applied to a small percentage of datasets. Subsequently, we trained and tested the accuracy of the model on adversarial datasets. We recorded the results for both benign and manipulated datasets and observed significant differences between the accuracy of the models on different datasets. From the experimental results, it is evident that the LF attack failed, whereas the FP attack showed effective results, which proved its significance in fooling a server. With a 1% LF attack on the CIC, the accuracy was approximately 0.0428 and the ASR was 0.9564; hence, the attack is easily detectable, while with a 1% FP attack, the accuracy and ASR were both approximately 0.9600, hence, FP attacks are difficult to detect. We repeated the experiment with different poisoning percentages.



## **15. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.00867v2) [paper-pdf](http://arxiv.org/pdf/2403.00867v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.



## **16. XAI-Based Detection of Adversarial Attacks on Deepfake Detectors**

cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02955v1) [paper-pdf](http://arxiv.org/pdf/2403.02955v1)

**Authors**: Ben Pinhasov, Raz Lapid, Rony Ohayon, Moshe Sipper, Yehudit Aperstein

**Abstract**: We introduce a novel methodology for identifying adversarial attacks on deepfake detectors using eXplainable Artificial Intelligence (XAI). In an era characterized by digital advancement, deepfakes have emerged as a potent tool, creating a demand for efficient detection systems. However, these systems are frequently targeted by adversarial attacks that inhibit their performance. We address this gap, developing a defensible deepfake detector by leveraging the power of XAI. The proposed methodology uses XAI to generate interpretability maps for a given method, providing explicit visualizations of decision-making factors within the AI models. We subsequently employ a pretrained feature extractor that processes both the input image and its corresponding XAI image. The feature embeddings extracted from this process are then used for training a simple yet effective classifier. Our approach contributes not only to the detection of deepfakes but also enhances the understanding of possible adversarial attacks, pinpointing potential vulnerabilities. Furthermore, this approach does not change the performance of the deepfake detector. The paper demonstrates promising results suggesting a potential pathway for future deepfake detection mechanisms. We believe this study will serve as a valuable contribution to the community, sparking much-needed discourse on safeguarding deepfake detectors.



## **17. Precise Extraction of Deep Learning Models via Side-Channel Attacks on Edge/Endpoint Devices**

cs.AI

Accepted by 27th European Symposium on Research in Computer Security  (ESORICS 2022)

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02870v1) [paper-pdf](http://arxiv.org/pdf/2403.02870v1)

**Authors**: Younghan Lee, Sohee Jun, Yungi Cho, Woorim Han, Hyungon Moon, Yunheung Paek

**Abstract**: With growing popularity, deep learning (DL) models are becoming larger-scale, and only the companies with vast training datasets and immense computing power can manage their business serving such large models. Most of those DL models are proprietary to the companies who thus strive to keep their private models safe from the model extraction attack (MEA), whose aim is to steal the model by training surrogate models. Nowadays, companies are inclined to offload the models from central servers to edge/endpoint devices. As revealed in the latest studies, adversaries exploit this opportunity as new attack vectors to launch side-channel attack (SCA) on the device running victim model and obtain various pieces of the model information, such as the model architecture (MA) and image dimension (ID). Our work provides a comprehensive understanding of such a relationship for the first time and would benefit future MEA studies in both offensive and defensive sides in that they may learn which pieces of information exposed by SCA are more important than the others. Our analysis additionally reveals that by grasping the victim model information from SCA, MEA can get highly effective and successful even without any prior knowledge of the model. Finally, to evince the practicality of our analysis results, we empirically apply SCA, and subsequently, carry out MEA under realistic threat assumptions. The results show up to 5.8 times better performance than when the adversary has no model information about the victim model.



## **18. FLGuard: Byzantine-Robust Federated Learning via Ensemble of Contrastive Models**

cs.LG

Accepted by 28th European Symposium on Research in Computer Security  (ESORICS 2023)

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02846v1) [paper-pdf](http://arxiv.org/pdf/2403.02846v1)

**Authors**: Younghan Lee, Yungi Cho, Woorim Han, Ho Bae, Yunheung Paek

**Abstract**: Federated Learning (FL) thrives in training a global model with numerous clients by only sharing the parameters of their local models trained with their private training datasets. Therefore, without revealing the private dataset, the clients can obtain a deep learning (DL) model with high performance. However, recent research proposed poisoning attacks that cause a catastrophic loss in the accuracy of the global model when adversaries, posed as benign clients, are present in a group of clients. Therefore, recent studies suggested byzantine-robust FL methods that allow the server to train an accurate global model even with the adversaries present in the system. However, many existing methods require the knowledge of the number of malicious clients or the auxiliary (clean) dataset or the effectiveness reportedly decreased hugely when the private dataset was non-independently and identically distributed (non-IID). In this work, we propose FLGuard, a novel byzantine-robust FL method that detects malicious clients and discards malicious local updates by utilizing the contrastive learning technique, which showed a tremendous improvement as a self-supervised learning method. With contrastive models, we design FLGuard as an ensemble scheme to maximize the defensive capability. We evaluate FLGuard extensively under various poisoning attacks and compare the accuracy of the global model with existing byzantine-robust FL methods. FLGuard outperforms the state-of-the-art defense methods in most cases and shows drastic improvement, especially in non-IID settings. https://github.com/201younghanlee/FLGuard



## **19. Here Comes The AI Worm: Unleashing Zero-click Worms that Target GenAI-Powered Applications**

cs.CR

Website: https://sites.google.com/view/compromptmized

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02817v1) [paper-pdf](http://arxiv.org/pdf/2403.02817v1)

**Authors**: Stav Cohen, Ron Bitton, Ben Nassi

**Abstract**: In the past year, numerous companies have incorporated Generative AI (GenAI) capabilities into new and existing applications, forming interconnected Generative AI (GenAI) ecosystems consisting of semi/fully autonomous agents powered by GenAI services. While ongoing research highlighted risks associated with the GenAI layer of agents (e.g., dialog poisoning, membership inference, prompt leaking, jailbreaking), a critical question emerges: Can attackers develop malware to exploit the GenAI component of an agent and launch cyber-attacks on the entire GenAI ecosystem? This paper introduces Morris II, the first worm designed to target GenAI ecosystems through the use of adversarial self-replicating prompts. The study demonstrates that attackers can insert such prompts into inputs that, when processed by GenAI models, prompt the model to replicate the input as output (replication), engaging in malicious activities (payload). Additionally, these inputs compel the agent to deliver them (propagate) to new agents by exploiting the connectivity within the GenAI ecosystem. We demonstrate the application of Morris II against GenAIpowered email assistants in two use cases (spamming and exfiltrating personal data), under two settings (black-box and white-box accesses), using two types of input data (text and images). The worm is tested against three different GenAI models (Gemini Pro, ChatGPT 4.0, and LLaVA), and various factors (e.g., propagation rate, replication, malicious activity) influencing the performance of the worm are evaluated.



## **20. Towards Robust Federated Learning via Logits Calibration on Non-IID Data**

cs.CV

Accepted by IEEE NOMS 2024

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02803v1) [paper-pdf](http://arxiv.org/pdf/2403.02803v1)

**Authors**: Yu Qiao, Apurba Adhikary, Chaoning Zhang, Choong Seon Hong

**Abstract**: Federated learning (FL) is a privacy-preserving distributed management framework based on collaborative model training of distributed devices in edge networks. However, recent studies have shown that FL is vulnerable to adversarial examples (AEs), leading to a significant drop in its performance. Meanwhile, the non-independent and identically distributed (non-IID) challenge of data distribution between edge devices can further degrade the performance of models. Consequently, both AEs and non-IID pose challenges to deploying robust learning models at the edge. In this work, we adopt the adversarial training (AT) framework to improve the robustness of FL models against adversarial example (AE) attacks, which can be termed as federated adversarial training (FAT). Moreover, we address the non-IID challenge by implementing a simple yet effective logits calibration strategy under the FAT framework, which can enhance the robustness of models when subjected to adversarial attacks. Specifically, we employ a direct strategy to adjust the logits output by assigning higher weights to classes with small samples during training. This approach effectively tackles the class imbalance in the training data, with the goal of mitigating biases between local and global models. Experimental results on three dataset benchmarks, MNIST, Fashion-MNIST, and CIFAR-10 show that our strategy achieves competitive results in natural and robust accuracy compared to several baselines.



## **21. On the Alignment of Group Fairness with Attribute Privacy**

cs.LG

arXiv admin note: text overlap with arXiv:2202.02242

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2211.10209v3) [paper-pdf](http://arxiv.org/pdf/2211.10209v3)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstract**: Group fairness and privacy are fundamental aspects in designing trustworthy machine learning models. Previous research has highlighted conflicts between group fairness and different privacy notions. We are the first to demonstrate the alignment of group fairness with the specific privacy notion of attribute privacy in a blackbox setting. Attribute privacy, quantified by the resistance to attribute inference attacks (AIAs), requires indistinguishability in the target model's output predictions. Group fairness guarantees this thereby mitigating AIAs and achieving attribute privacy. To demonstrate this, we first introduce AdaptAIA, an enhancement of existing AIAs, tailored for real-world datasets with class imbalances in sensitive attributes. Through theoretical and extensive empirical analyses, we demonstrate the efficacy of two standard group fairness algorithms (i.e., adversarial debiasing and exponentiated gradient descent) against AdaptAIA. Additionally, since using group fairness results in attribute privacy, it acts as a defense against AIAs, which is currently lacking. Overall, we show that group fairness aligns with attribute privacy at no additional cost other than the already existing trade-off with model utility.



## **22. Minimum Topology Attacks for Graph Neural Networks**

cs.AI

Published on WWW 2023. Proceedings of the ACM Web Conference 2023

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02723v1) [paper-pdf](http://arxiv.org/pdf/2403.02723v1)

**Authors**: Mengmei Zhang, Xiao Wang, Chuan Shi, Lingjuan Lyu, Tianchi Yang, Junping Du

**Abstract**: With the great popularity of Graph Neural Networks (GNNs), their robustness to adversarial topology attacks has received significant attention. Although many attack methods have been proposed, they mainly focus on fixed-budget attacks, aiming at finding the most adversarial perturbations within a fixed budget for target node. However, considering the varied robustness of each node, there is an inevitable dilemma caused by the fixed budget, i.e., no successful perturbation is found when the budget is relatively small, while if it is too large, the yielding redundant perturbations will hurt the invisibility. To break this dilemma, we propose a new type of topology attack, named minimum-budget topology attack, aiming to adaptively find the minimum perturbation sufficient for a successful attack on each node. To this end, we propose an attack model, named MiBTack, based on a dynamic projected gradient descent algorithm, which can effectively solve the involving non-convex constraint optimization on discrete topology. Extensive results on three GNNs and four real-world datasets show that MiBTack can successfully lead all target nodes misclassified with the minimum perturbation edges. Moreover, the obtained minimum budget can be used to measure node robustness, so we can explore the relationships of robustness, topology, and uncertainty for nodes, which is beyond what the current fixed-budget topology attacks can offer.



## **23. ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**

cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2312.03085v2) [paper-pdf](http://arxiv.org/pdf/2312.03085v2)

**Authors**: Xiaohu Lu, Hayder Radha

**Abstract**: The adversarial robustness of a model is its ability to resist adversarial attacks in the form of small perturbations to input data. Universal adversarial attack methods such as Fast Sign Gradient Method (FSGM) and Projected Gradient Descend (PGD) are popular for LiDAR object detection, but they are often deficient compared to task-specific adversarial attacks. Additionally, these universal methods typically require unrestricted access to the model's information, which is difficult to obtain in real-world applications. To address these limitations, we present a black-box Scaling Adversarial Robustness (ScAR) method for LiDAR object detection. By analyzing the statistical characteristics of 3D object detection datasets such as KITTI, Waymo, and nuScenes, we have found that the model's prediction is sensitive to scaling of 3D instances. We propose three black-box scaling adversarial attack methods based on the available information: model-aware attack, distribution-aware attack, and blind attack. We also introduce a strategy for generating scaling adversarial examples to improve the model's robustness against these three scaling adversarial attacks. Comparison with other methods on public datasets under different 3D object detection architectures demonstrates the effectiveness of our proposed method. Our code is available at https://github.com/xiaohulugo/ScAR-IROS2023.



## **24. Towards Poisoning Fair Representations**

cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2309.16487v2) [paper-pdf](http://arxiv.org/pdf/2309.16487v2)

**Authors**: Tianci Liu, Haoyu Wang, Feijie Wu, Hengtong Zhang, Pan Li, Lu Su, Jing Gao

**Abstract**: Fair machine learning seeks to mitigate model prediction bias against certain demographic subgroups such as elder and female. Recently, fair representation learning (FRL) trained by deep neural networks has demonstrated superior performance, whereby representations containing no demographic information are inferred from the data and then used as the input to classification or other downstream tasks. Despite the development of FRL methods, their vulnerability under data poisoning attack, a popular protocol to benchmark model robustness under adversarial scenarios, is under-explored. Data poisoning attacks have been developed for classical fair machine learning methods which incorporate fairness constraints into shallow-model classifiers. Nonetheless, these attacks fall short in FRL due to notably different fairness goals and model architectures. This work proposes the first data poisoning framework attacking FRL. We induce the model to output unfair representations that contain as much demographic information as possible by injecting carefully crafted poisoning samples into the training data. This attack entails a prohibitive bilevel optimization, wherefore an effective approximated solution is proposed. A theoretical analysis on the needed number of poisoning samples is derived and sheds light on defending against the attack. Experiments on benchmark fairness datasets and state-of-the-art fair representation learning models demonstrate the superiority of our attack.



## **25. COMMIT: Certifying Robustness of Multi-Sensor Fusion Systems against Semantic Attacks**

cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02329v1) [paper-pdf](http://arxiv.org/pdf/2403.02329v1)

**Authors**: Zijian Huang, Wenda Chu, Linyi Li, Chejian Xu, Bo Li

**Abstract**: Multi-sensor fusion systems (MSFs) play a vital role as the perception module in modern autonomous vehicles (AVs). Therefore, ensuring their robustness against common and realistic adversarial semantic transformations, such as rotation and shifting in the physical world, is crucial for the safety of AVs. While empirical evidence suggests that MSFs exhibit improved robustness compared to single-modal models, they are still vulnerable to adversarial semantic transformations. Despite the proposal of empirical defenses, several works show that these defenses can be attacked again by new adaptive attacks. So far, there is no certified defense proposed for MSFs. In this work, we propose the first robustness certification framework COMMIT certify robustness of multi-sensor fusion systems against semantic attacks. In particular, we propose a practical anisotropic noise mechanism that leverages randomized smoothing with multi-modal data and performs a grid-based splitting method to characterize complex semantic transformations. We also propose efficient algorithms to compute the certification in terms of object detection accuracy and IoU for large-scale MSF models. Empirically, we evaluate the efficacy of COMMIT in different settings and provide a comprehensive benchmark of certified robustness for different MSF models using the CARLA simulation platform. We show that the certification for MSF models is at most 48.39% higher than that of single-modal models, which validates the advantages of MSF models. We believe our certification framework and benchmark will contribute an important step towards certifiably robust AVs in practice.



## **26. Mirage: Defense against CrossPath Attacks in Software Defined Networks**

cs.CR

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02172v1) [paper-pdf](http://arxiv.org/pdf/2403.02172v1)

**Authors**: Shariq Murtuza, Krishna Asawa

**Abstract**: The Software-Defined Networks (SDNs) face persistent threats from various adversaries that attack them using different methods to mount Denial of Service attacks. These attackers have different motives and follow diverse tactics to achieve their nefarious objectives. In this work, we focus on the impact of CrossPath attacks in SDNs and introduce our framework, Mirage, which not only detects but also mitigates this attack. Our framework, Mirage, detects SDN switches that become unreachable due to being under attack, takes proactive measures to prevent Adversarial Path Reconnaissance, and effectively mitigates CrossPath attacks in SDNs. A CrossPath attack is a form of link flood attack that indirectly attacks the control plane by overwhelming the shared links that connect the data and control planes with data plane traffic. This attack is exclusive to in band SDN, where the data and the control plane, both utilize the same physical links for transmitting and receiving traffic. Our framework, Mirage, prevents attackers from launching adversarial path reconnaissance to identify shared links in a network, thereby thwarting their abuse and preventing this attack. Mirage not only stops adversarial path reconnaissance but also includes features to quickly counter ongoing attacks once detected. Mirage uses path diversity to reroute network packet to prevent timing based measurement. Mirage can also enforce short lived flow table rules to prevent timing attacks. These measures are carefully designed to enhance the security of the SDN environment. Moreover, we share the results of our experiments, which clearly show Mirage's effectiveness in preventing path reconnaissance, detecting CrossPath attacks, and mitigating ongoing threats. Our framework successfully protects the network from these harmful activities, giving valuable insights into SDN security.



## **27. Rethinking Model Ensemble in Transfer-based Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2303.09105v2) [paper-pdf](http://arxiv.org/pdf/2303.09105v2)

**Authors**: Huanran Chen, Yichi Zhang, Yinpeng Dong, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: It is widely recognized that deep learning models lack robustness to adversarial examples. An intriguing property of adversarial examples is that they can transfer across different models, which enables black-box attacks without any knowledge of the victim model. An effective strategy to improve the transferability is attacking an ensemble of models. However, previous works simply average the outputs of different models, lacking an in-depth analysis on how and why model ensemble methods can strongly improve the transferability. In this paper, we rethink the ensemble in adversarial attacks and define the common weakness of model ensemble with two properties: 1) the flatness of loss landscape; and 2) the closeness to the local optimum of each model. We empirically and theoretically show that both properties are strongly correlated with the transferability and propose a Common Weakness Attack (CWA) to generate more transferable adversarial examples by promoting these two properties. Experimental results on both image classification and object detection tasks validate the effectiveness of our approach to improving the adversarial transferability, especially when attacking adversarially trained models. We also successfully apply our method to attack a black-box large vision-language model -- Google's Bard, showing the practical effectiveness. Code is available at \url{https://github.com/huanranchen/AdversarialAttacks}.



## **28. Robustness Bounds on the Successful Adversarial Examples: Theory and Practice**

cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01896v1) [paper-pdf](http://arxiv.org/pdf/2403.01896v1)

**Authors**: Hiroaki Maeshima, Akira Otsuka

**Abstract**: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification. We proved a new upper bound that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.



## **29. One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models**

cs.CV

CVPR2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01849v1) [paper-pdf](http://arxiv.org/pdf/2403.01849v1)

**Authors**: Lin Li, Haoyan Guan, Jianing Qiu, Michael Spratling

**Abstract**: Large pre-trained Vision-Language Models (VLMs) like CLIP, despite having remarkable generalization ability, are highly vulnerable to adversarial examples. This work studies the adversarial robustness of VLMs from the novel perspective of the text prompt instead of the extensively studied model weights (frozen in this work). We first show that the effectiveness of both adversarial attack and defense are sensitive to the used text prompt. Inspired by this, we propose a method to improve resilience to adversarial attacks by learning a robust text prompt for VLMs. The proposed method, named Adversarial Prompt Tuning (APT), is effective while being both computationally and data efficient. Extensive experiments are conducted across 15 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show APT's superiority over hand-engineered prompts and other state-of-the-art adaption methods. APT demonstrated excellent abilities in terms of the in-distribution performance and the generalization under input distribution shift and across datasets. Surprisingly, by simply adding one learned word to the prompts, APT can significantly boost the accuracy and robustness (epsilon=4/255) over the hand-engineered prompts by +13% and +8.5% on average respectively. The improvement further increases, in our most effective setting, to +26.4% for accuracy and +16.7% for robustness. Code is available at https://github.com/TreeLLi/APT.



## **30. Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: A Robustness Study**

cs.LG

in Proc. AAAI Conference on Artificial Intelligence, Vancouver,  Canada, Feb. 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2401.04331v2) [paper-pdf](http://arxiv.org/pdf/2401.04331v2)

**Authors**: Qiyu Kang, Kai Zhao, Yang Song, Yihang Xie, Yanan Zhao, Sijie Wang, Rui She, Wee Peng Tay

**Abstract**: In this work, we rigorously investigate the robustness of graph neural fractional-order differential equation (FDE) models. This framework extends beyond traditional graph neural (integer-order) ordinary differential equation (ODE) models by implementing the time-fractional Caputo derivative. Utilizing fractional calculus allows our model to consider long-term memory during the feature updating process, diverging from the memoryless Markovian updates seen in traditional graph neural ODE models. The superiority of graph neural FDE models over graph neural ODE models has been established in environments free from attacks or perturbations. While traditional graph neural ODE models have been verified to possess a degree of stability and resilience in the presence of adversarial attacks in existing literature, the robustness of graph neural FDE models, especially under adversarial conditions, remains largely unexplored. This paper undertakes a detailed assessment of the robustness of graph neural FDE models. We establish a theoretical foundation outlining the robustness characteristics of graph neural FDE models, highlighting that they maintain more stringent output perturbation bounds in the face of input and graph topology disturbances, compared to their integer-order counterparts. Our empirical evaluations further confirm the enhanced robustness of graph neural FDE models, highlighting their potential in adversarially robust applications.



## **31. LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper**

cs.CR

Fixed the bibliography reference issue in our LLM jailbreak defense  vision paper submitted on 24 Feb 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.15727v2) [paper-pdf](http://arxiv.org/pdf/2402.15727v2)

**Authors**: Daoyuan Wu, Shuai Wang, Yang Liu, Ning Liu

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs). A considerable amount of research exists proposing more effective jailbreak attacks, including the recent Greedy Coordinate Gradient (GCG) attack, jailbreak template-based attacks such as using "Do-Anything-Now" (DAN), and multilingual jailbreak. In contrast, the defensive side has been relatively less explored. This paper proposes a lightweight yet practical defense called SELFDEFEND, which can defend against all existing jailbreak attacks with minimal delay for jailbreak prompts and negligible delay for normal user prompts. Our key insight is that regardless of the kind of jailbreak strategies employed, they eventually need to include a harmful prompt (e.g., "how to make a bomb") in the prompt sent to LLMs, and we found that existing LLMs can effectively recognize such harmful prompts that violate their safety policies. Based on this insight, we design a shadow stack that concurrently checks whether a harmful prompt exists in the user prompt and triggers a checkpoint in the normal stack once a token of "No" or a harmful prompt is output. The latter could also generate an explainable LLM response to adversarial prompts. We demonstrate our idea of SELFDEFEND works in various jailbreak scenarios through manual analysis in GPT-3.5/4. We also list three future directions to further enhance SELFDEFEND.



## **32. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.18607v2) [paper-pdf](http://arxiv.org/pdf/2402.18607v2)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.



## **33. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

Accepted at AISTATS 2024, a preliminary version appeared at ICML 2023  AdvML Workshop

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2008.09312v7) [paper-pdf](http://arxiv.org/pdf/2008.09312v7)

**Authors**: Shiliang Zuo

**Abstract**: I study adversarial attacks against stochastic bandit algorithms. At each round, the learner chooses an arm, and a stochastic reward is generated. The adversary strategically adds corruption to the reward, and the learner is only able to observe the corrupted reward at each round. Two sets of results are presented in this paper. The first set studies the optimal attack strategies for the adversary. The adversary has a target arm he wishes to promote, and his goal is to manipulate the learner into choosing this target arm $T - o(T)$ times. I design attack strategies against UCB and Thompson Sampling that only spends $\widehat{O}(\sqrt{\log T})$ cost. Matching lower bounds are presented, and the vulnerability of UCB, Thompson sampling and $\varepsilon$-greedy are exactly characterized. The second set studies how the learner can defend against the adversary. Inspired by literature on smoothed analysis and behavioral economics, I present two simple algorithms that achieve a competitive ratio arbitrarily close to 1.



## **34. Adversarial Attacks on Fairness of Graph Neural Networks**

cs.LG

Accepted at ICLR 2024

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2310.13822v2) [paper-pdf](http://arxiv.org/pdf/2310.13822v2)

**Authors**: Binchi Zhang, Yushun Dong, Chen Chen, Yada Zhu, Minnan Luo, Jundong Li

**Abstract**: Fairness-aware graph neural networks (GNNs) have gained a surge of attention as they can reduce the bias of predictions on any demographic group (e.g., female) in graph-based applications. Although these methods greatly improve the algorithmic fairness of GNNs, the fairness can be easily corrupted by carefully designed adversarial attacks. In this paper, we investigate the problem of adversarial attacks on fairness of GNNs and propose G-FairAttack, a general framework for attacking various types of fairness-aware GNNs in terms of fairness with an unnoticeable effect on prediction utility. In addition, we propose a fast computation technique to reduce the time complexity of G-FairAttack. The experimental study demonstrates that G-FairAttack successfully corrupts the fairness of different types of GNNs while keeping the attack unnoticeable. Our study on fairness attacks sheds light on potential vulnerabilities in fairness-aware GNNs and guides further research on the robustness of GNNs in terms of fairness.



## **35. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2311.16119v3) [paper-pdf](http://arxiv.org/pdf/2311.16119v3)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.



## **36. Gradient Shaping: Enhancing Backdoor Attack Against Reverse Engineering**

cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2301.12318v2) [paper-pdf](http://arxiv.org/pdf/2301.12318v2)

**Authors**: Rui Zhu, Di Tang, Siyuan Tang, Guanhong Tao, Shiqing Ma, Xiaofeng Wang, Haixu Tang

**Abstract**: Most existing methods to detect backdoored machine learning (ML) models take one of the two approaches: trigger inversion (aka. reverse engineer) and weight analysis (aka. model diagnosis). In particular, the gradient-based trigger inversion is considered to be among the most effective backdoor detection techniques, as evidenced by the TrojAI competition, Trojan Detection Challenge and backdoorBench. However, little has been done to understand why this technique works so well and, more importantly, whether it raises the bar to the backdoor attack. In this paper, we report the first attempt to answer this question by analyzing the change rate of the backdoored model around its trigger-carrying inputs. Our study shows that existing attacks tend to inject the backdoor characterized by a low change rate around trigger-carrying inputs, which are easy to capture by gradient-based trigger inversion. In the meantime, we found that the low change rate is not necessary for a backdoor attack to succeed: we design a new attack enhancement called \textit{Gradient Shaping} (GRASP), which follows the opposite direction of adversarial training to reduce the change rate of a backdoored model with regard to the trigger, without undermining its backdoor effect. Also, we provide a theoretic analysis to explain the effectiveness of this new technique and the fundamental weakness of gradient-based trigger inversion. Finally, we perform both theoretical and experimental analysis, showing that the GRASP enhancement does not reduce the effectiveness of the stealthy attacks against the backdoor detection methods based on weight analysis, as well as other backdoor mitigation methods without using detection.



## **37. Fusion is Not Enough: Single Modal Attacks on Fusion Models for 3D Object Detection**

cs.CV

Accepted at ICLR'2024

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2304.14614v3) [paper-pdf](http://arxiv.org/pdf/2304.14614v3)

**Authors**: Zhiyuan Cheng, Hongjun Choi, James Liang, Shiwei Feng, Guanhong Tao, Dongfang Liu, Michael Zuzak, Xiangyu Zhang

**Abstract**: Multi-sensor fusion (MSF) is widely used in autonomous vehicles (AVs) for perception, particularly for 3D object detection with camera and LiDAR sensors. The purpose of fusion is to capitalize on the advantages of each modality while minimizing its weaknesses. Advanced deep neural network (DNN)-based fusion techniques have demonstrated the exceptional and industry-leading performance. Due to the redundant information in multiple modalities, MSF is also recognized as a general defence strategy against adversarial attacks. In this paper, we attack fusion models from the camera modality that is considered to be of lesser importance in fusion but is more affordable for attackers. We argue that the weakest link of fusion models depends on their most vulnerable modality, and propose an attack framework that targets advanced camera-LiDAR fusion-based 3D object detection models through camera-only adversarial attacks. Our approach employs a two-stage optimization-based strategy that first thoroughly evaluates vulnerable image areas under adversarial attacks, and then applies dedicated attack strategies for different fusion models to generate deployable patches. The evaluations with six advanced camera-LiDAR fusion models and one camera-only model indicate that our attacks successfully compromise all of them. Our approach can either decrease the mean average precision (mAP) of detection performance from 0.824 to 0.353, or degrade the detection score of a target object from 0.728 to 0.156, demonstrating the efficacy of our proposed attack framework. Code is available.



## **38. Accelerating Greedy Coordinate Gradient via Probe Sampling**

cs.CL

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01251v1) [paper-pdf](http://arxiv.org/pdf/2403.01251v1)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a central issue given their rapid progress and wide applications. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing prompts containing adversarial suffixes to break the presumingly safe LLMs, but the optimization of GCG is time-consuming and limits its practicality. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$ to accelerate the GCG algorithm. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates to reduce the computation time. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b and leads to equal or improved attack success rate (ASR) on the AdvBench.



## **39. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

cs.LG

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01218v1) [paper-pdf](http://arxiv.org/pdf/2403.01218v1)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their ``U-MIA'' counterparts). We propose a categorization of existing U-MIAs into ``population U-MIAs'', where the same attacker is instantiated for all examples, and ``per-example U-MIAs'', where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.



## **40. SAR-AE-SFP: SAR Imagery Adversarial Example in Real Physics domain with Target Scattering Feature Parameters**

cs.CV

10 pages, 9 figures, 2 tables

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01210v1) [paper-pdf](http://arxiv.org/pdf/2403.01210v1)

**Authors**: Jiahao Cui, Jiale Duan, Binyan Luo, Hang Cao, Wang Guo, Haifeng Li

**Abstract**: Deep neural network-based Synthetic Aperture Radar (SAR) target recognition models are susceptible to adversarial examples. Current adversarial example generation methods for SAR imagery primarily operate in the 2D digital domain, known as image adversarial examples. Recent work, while considering SAR imaging scatter mechanisms, fails to account for the actual imaging process, rendering attacks in the three-dimensional physical domain infeasible, termed pseudo physics adversarial examples. To address these challenges, this paper proposes SAR-AE-SFP-Attack, a method to generate real physics adversarial examples by altering the scattering feature parameters of target objects. Specifically, we iteratively optimize the coherent energy accumulation of the target echo by perturbing the reflection coefficient and scattering coefficient in the scattering feature parameters of the three-dimensional target object, and obtain the adversarial example after echo signal processing and imaging processing in the RaySAR simulator. Experimental results show that compared to digital adversarial attack methods, SAR-AE-SFP Attack significantly improves attack efficiency on CNN-based models (over 30\%) and Transformer-based models (over 13\%), demonstrating significant transferability of attack effects across different models and perspectives.



## **41. Harnessing the Speed and Accuracy of Machine Learning to Advance Cybersecurity**

cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2302.12415v3) [paper-pdf](http://arxiv.org/pdf/2302.12415v3)

**Authors**: Khatoon Mohammed

**Abstract**: As cyber attacks continue to increase in frequency and sophistication, detecting malware has become a critical task for maintaining the security of computer systems. Traditional signature-based methods of malware detection have limitations in detecting complex and evolving threats. In recent years, machine learning (ML) has emerged as a promising solution to detect malware effectively. ML algorithms are capable of analyzing large datasets and identifying patterns that are difficult for humans to identify. This paper presents a comprehensive review of the state-of-the-art ML techniques used in malware detection, including supervised and unsupervised learning, deep learning, and reinforcement learning. We also examine the challenges and limitations of ML-based malware detection, such as the potential for adversarial attacks and the need for large amounts of labeled data. Furthermore, we discuss future directions in ML-based malware detection, including the integration of multiple ML algorithms and the use of explainable AI techniques to enhance the interpret ability of ML-based detection systems. Our research highlights the potential of ML-based techniques to improve the speed and accuracy of malware detection, and contribute to enhancing cybersecurity



## **42. False Claims against Model Ownership Resolution**

cs.CR

13pages,3 figures. To appear in the 33rd USENIX Security Symposium  (USENIX Security '24)

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2304.06607v5) [paper-pdf](http://arxiv.org/pdf/2304.06607v5)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation, we demonstrate that our false claim attacks always succeed in the MOR schemes that follow our generalization, including against a real-world model: Amazon's Rekognition API.



## **43. Self-Guided Robust Graph Structure Refinement**

cs.LG

This paper has been accepted by TheWebConf 2024 (Oral Presentation)

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2402.11837v2) [paper-pdf](http://arxiv.org/pdf/2402.11837v2)

**Authors**: Yeonjun In, Kanghoon Yoon, Kibum Kim, Kijung Shin, Chanyoung Park

**Abstract**: Recent studies have revealed that GNNs are vulnerable to adversarial attacks. To defend against such attacks, robust graph structure refinement (GSR) methods aim at minimizing the effect of adversarial edges based on node features, graph structure, or external information. However, we have discovered that existing GSR methods are limited by narrowassumptions, such as assuming clean node features, moderate structural attacks, and the availability of external clean graphs, resulting in the restricted applicability in real-world scenarios. In this paper, we propose a self-guided GSR framework (SG-GSR), which utilizes a clean sub-graph found within the given attacked graph itself. Furthermore, we propose a novel graph augmentation and a group-training strategy to handle the two technical challenges in the clean sub-graph extraction: 1) loss of structural information, and 2) imbalanced node degree distribution. Extensive experiments demonstrate the effectiveness of SG-GSR under various scenarios including non-targeted attacks, targeted attacks, feature attacks, e-commerce fraud, and noisy node labels. Our code is available at https://github.com/yeonjun-in/torch-SG-GSR.



## **44. Attacking the Diebold Signature Variant -- RSA Signatures with Unverified High-order Padding**

cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01048v1) [paper-pdf](http://arxiv.org/pdf/2403.01048v1)

**Authors**: Ryan W. Gardner, Tadayoshi Kohno, Alec Yasinsac

**Abstract**: We examine a natural but improper implementation of RSA signature verification deployed on the widely used Diebold Touch Screen and Optical Scan voting machines. In the implemented scheme, the verifier fails to examine a large number of the high-order bits of signature padding and the public exponent is three. We present an very mathematically simple attack that enables an adversary to forge signatures on arbitrary messages in a negligible amount of time.



## **45. Resilience of Entropy Model in Distributed Neural Networks**

cs.LG

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00942v1) [paper-pdf](http://arxiv.org/pdf/2403.00942v1)

**Authors**: Milin Zhang, Mohammad Abdi, Shahriar Rifat, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have emerged as a key technique to reduce communication overhead without sacrificing performance in edge computing systems. Recently, entropy coding has been introduced to further reduce the communication overhead. The key idea is to train the distributed DNN jointly with an entropy model, which is used as side information during inference time to adaptively encode latent representations into bit streams with variable length. To the best of our knowledge, the resilience of entropy models is yet to be investigated. As such, in this paper we formulate and investigate the resilience of entropy models to intentional interference (e.g., adversarial attacks) and unintentional interference (e.g., weather changes and motion blur). Through an extensive experimental campaign with 3 different DNN architectures, 2 entropy models and 4 rate-distortion trade-off factors, we demonstrate that the entropy attacks can increase the communication overhead by up to 95%. By separating compression features in frequency and spatial domain, we propose a new defense mechanism that can reduce the transmission overhead of the attacked input by about 9% compared to unperturbed data, with only about 2% accuracy loss. Importantly, the proposed defense mechanism is a standalone approach which can be applied in conjunction with approaches such as adversarial training to further improve robustness. Code will be shared for reproducibility.



## **46. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

cs.CV

under review

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2401.06637v4) [paper-pdf](http://arxiv.org/pdf/2401.06637v4)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.



## **47. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

cs.CV

accepted at VISAPP23

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2212.06776v5) [paper-pdf](http://arxiv.org/pdf/2212.06776v5)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID



## **48. Protect and Extend -- Using GANs for Synthetic Data Generation of Time-Series Medical Records**

cs.LG

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.14042v2) [paper-pdf](http://arxiv.org/pdf/2402.14042v2)

**Authors**: Navid Ashrafi, Vera Schmitt, Robert P. Spang, Sebastian Möller, Jan-Niklas Voigt-Antons

**Abstract**: Preservation of private user data is of paramount importance for high Quality of Experience (QoE) and acceptability, particularly with services treating sensitive data, such as IT-based health services. Whereas anonymization techniques were shown to be prone to data re-identification, synthetic data generation has gradually replaced anonymization since it is relatively less time and resource-consuming and more robust to data leakage. Generative Adversarial Networks (GANs) have been used for generating synthetic datasets, especially GAN frameworks adhering to the differential privacy phenomena. This research compares state-of-the-art GAN-based models for synthetic data generation to generate time-series synthetic medical records of dementia patients which can be distributed without privacy concerns. Predictive modeling, autocorrelation, and distribution analysis are used to assess the Quality of Generating (QoG) of the generated data. The privacy preservation of the respective models is assessed by applying membership inference attacks to determine potential data leakage risks. Our experiments indicate the superiority of the privacy-preserving GAN (PPGAN) model over other models regarding privacy preservation while maintaining an acceptable level of QoG. The presented results can support better data protection for medical use cases in the future.



## **49. Attacking Delay-based PUFs with Minimal Adversary Model**

cs.CR

13 pages, 6 figures, journal

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00464v1) [paper-pdf](http://arxiv.org/pdf/2403.00464v1)

**Authors**: Hongming Fei, Owen Millwood, Prosanta Gope, Jack Miskelly, Biplab Sikdar

**Abstract**: Physically Unclonable Functions (PUFs) provide a streamlined solution for lightweight device authentication. Delay-based Arbiter PUFs, with their ease of implementation and vast challenge space, have received significant attention; however, they are not immune to modelling attacks that exploit correlations between their inputs and outputs. Research is therefore polarized between developing modelling-resistant PUFs and devising machine learning attacks against them. This dichotomy often results in exaggerated concerns and overconfidence in PUF security, primarily because there lacks a universal tool to gauge a PUF's security. In many scenarios, attacks require additional information, such as PUF type or configuration parameters. Alarmingly, new PUFs are often branded `secure' if they lack a specific attack model upon introduction. To impartially assess the security of delay-based PUFs, we present a generic framework featuring a Mixture-of-PUF-Experts (MoPE) structure for mounting attacks on various PUFs with minimal adversarial knowledge, which provides a way to compare their performance fairly and impartially. We demonstrate the capability of our model to attack different PUF types, including the first successful attack on Heterogeneous Feed-Forward PUFs using only a reasonable amount of challenges and responses. We propose an extension version of our model, a Multi-gate Mixture-of-PUF-Experts (MMoPE) structure, facilitating multi-task learning across diverse PUFs to recognise commonalities across PUF designs. This allows a streamlining of training periods for attacking multiple PUFs simultaneously. We conclude by showcasing the potent performance of MoPE and MMoPE across a spectrum of PUF types, employing simulated, real-world unbiased, and biased data sets for analysis.



## **50. Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey**

cs.LG

57 pages, 16 figues, 2 tables

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00420v1) [paper-pdf](http://arxiv.org/pdf/2403.00420v1)

**Authors**: Lucas Schott, Josephine Delas, Hatem Hajri, Elies Gherbi, Reda Yaich, Nora Boulahia-Cuppens, Frederic Cuppens, Sylvain Lamprier

**Abstract**: Deep Reinforcement Learning (DRL) is an approach for training autonomous agents across various complex environments. Despite its significant performance in well known environments, it remains susceptible to minor conditions variations, raising concerns about its reliability in real-world applications. To improve usability, DRL must demonstrate trustworthiness and robustness. A way to improve robustness of DRL to unknown changes in the conditions is through Adversarial Training, by training the agent against well suited adversarial attacks on the dynamics of the environment. Addressing this critical issue, our work presents an in-depth analysis of contemporary adversarial attack methodologies, systematically categorizing them and comparing their objectives and operational mechanisms. This classification offers a detailed insight into how adversarial attacks effectively act for evaluating the resilience of DRL agents, thereby paving the way for enhancing their robustness.



