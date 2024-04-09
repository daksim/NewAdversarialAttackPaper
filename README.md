# Latest Adversarial Attack Papers
**update at 2024-04-09 20:03:35**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Case Study: Neural Network Malware Detection Verification for Feature and Image Datasets**

cs.CR

In International Conference On Formal Methods in Software  Engineering, 2024; (FormaliSE'24)

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05703v1) [paper-pdf](http://arxiv.org/pdf/2404.05703v1)

**Authors**: Preston K. Robinette, Diego Manzanas Lopez, Serena Serbinowska, Kevin Leach, Taylor T. Johnson

**Abstract**: Malware, or software designed with harmful intent, is an ever-evolving threat that can have drastic effects on both individuals and institutions. Neural network malware classification systems are key tools for combating these threats but are vulnerable to adversarial machine learning attacks. These attacks perturb input data to cause misclassification, bypassing protective systems. Existing defenses often rely on enhancing the training process, thereby increasing the model's robustness to these perturbations, which is quantified using verification. While training improvements are necessary, we propose focusing on the verification process used to evaluate improvements to training. As such, we present a case study that evaluates a novel verification domain that will help to ensure tangible safeguards against adversaries and provide a more reliable means of evaluating the robustness and effectiveness of anti-malware systems. To do so, we describe malware classification and two types of common malware datasets (feature and image datasets), demonstrate the certified robustness accuracy of malware classifiers using the Neural Network Verification (NNV) and Neural Network Enumeration (nnenum) tools, and outline the challenges and future considerations necessary for the improvement and refinement of the verification of malware classification. By evaluating this novel domain as a case study, we hope to increase its visibility, encourage further research and scrutiny, and ultimately enhance the resilience of digital systems against malicious attacks.



## **2. David and Goliath: An Empirical Evaluation of Attacks and Defenses for QNNs at the Deep Edge**

cs.LG

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05688v1) [paper-pdf](http://arxiv.org/pdf/2404.05688v1)

**Authors**: Miguel Costa, Sandro Pinto

**Abstract**: ML is shifting from the cloud to the edge. Edge computing reduces the surface exposing private data and enables reliable throughput guarantees in real-time applications. Of the panoply of devices deployed at the edge, resource-constrained MCUs, e.g., Arm Cortex-M, are more prevalent, orders of magnitude cheaper, and less power-hungry than application processors or GPUs. Thus, enabling intelligence at the deep edge is the zeitgeist, with researchers focusing on unveiling novel approaches to deploy ANNs on these constrained devices. Quantization is a well-established technique that has proved effective in enabling the deployment of neural networks on MCUs; however, it is still an open question to understand the robustness of QNNs in the face of adversarial examples.   To fill this gap, we empirically evaluate the effectiveness of attacks and defenses from (full-precision) ANNs on (constrained) QNNs. Our evaluation includes three QNNs targeting TinyML applications, ten attacks, and six defenses. With this study, we draw a set of interesting findings. First, quantization increases the point distance to the decision boundary and leads the gradient estimated by some attacks to explode or vanish. Second, quantization can act as a noise attenuator or amplifier, depending on the noise magnitude, and causes gradient misalignment. Regarding adversarial defenses, we conclude that input pre-processing defenses show impressive results on small perturbations; however, they fall short as the perturbation increases. At the same time, train-based defenses increase the average point distance to the decision boundary, which holds after quantization. However, we argue that train-based defenses still need to smooth the quantization-shift and gradient misalignment phenomenons to counteract adversarial example transferability to QNNs. All artifacts are open-sourced to enable independent validation of results.



## **3. Investigating the Impact of Quantization on Adversarial Robustness**

cs.LG

Accepted to ICLR 2024 Workshop PML4LRS

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05639v1) [paper-pdf](http://arxiv.org/pdf/2404.05639v1)

**Authors**: Qun Li, Yuan Meng, Chen Tang, Jiacheng Jiang, Zhi Wang

**Abstract**: Quantization is a promising technique for reducing the bit-width of deep models to improve their runtime performance and storage efficiency, and thus becomes a fundamental step for deployment. In real-world scenarios, quantized models are often faced with adversarial attacks which cause the model to make incorrect inferences by introducing slight perturbations. However, recent studies have paid less attention to the impact of quantization on the model robustness. More surprisingly, existing studies on this topic even present inconsistent conclusions, which prompted our in-depth investigation. In this paper, we conduct a first-time analysis of the impact of the quantization pipeline components that can incorporate robust optimization under the settings of Post-Training Quantization and Quantization-Aware Training. Through our detailed analysis, we discovered that this inconsistency arises from the use of different pipelines in different studies, specifically regarding whether robust optimization is performed and at which quantization stage it occurs. Our research findings contribute insights into deploying more secure and robust quantized networks, assisting practitioners in reference for scenarios with high-security requirements and limited resources.



## **4. SoK: Gradient Leakage in Federated Learning**

cs.CR

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05403v1) [paper-pdf](http://arxiv.org/pdf/2404.05403v1)

**Authors**: Jiacheng Du, Jiahui Hu, Zhibo Wang, Peng Sun, Neil Zhenqiang Gong, Kui Ren

**Abstract**: Federated learning (FL) enables collaborative model training among multiple clients without raw data exposure. However, recent studies have shown that clients' private training data can be reconstructed from the gradients they share in FL, known as gradient inversion attacks (GIAs). While GIAs have demonstrated effectiveness under \emph{ideal settings and auxiliary assumptions}, their actual efficacy against \emph{practical FL systems} remains under-explored. To address this gap, we conduct a comprehensive study on GIAs in this work. We start with a survey of GIAs that establishes a milestone to trace their evolution and develops a systematization to uncover their inherent threats. Specifically, we categorize the auxiliary assumptions used by existing GIAs based on their practical accessibility to potential adversaries. To facilitate deeper analysis, we highlight the challenges that GIAs face in practical FL systems from three perspectives: \textit{local training}, \textit{model}, and \textit{post-processing}. We then perform extensive theoretical and empirical evaluations of state-of-the-art GIAs across diverse settings, utilizing eight datasets and thirteen models. Our findings indicate that GIAs have inherent limitations when reconstructing data under practical local training settings. Furthermore, their efficacy is sensitive to the trained model, and even simple post-processing measures applied to gradients can be effective defenses. Overall, our work provides crucial insights into the limited effectiveness of GIAs in practical FL systems. By rectifying prior misconceptions, we hope to inspire more accurate and realistic investigations on this topic.



## **5. BruSLeAttack: A Query-Efficient Score-Based Black-Box Sparse Adversarial Attack**

cs.LG

Published as a conference paper at the International Conference on  Learning Representations (ICLR 2024). Code is available at  https://brusliattack.github.io/

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05311v1) [paper-pdf](http://arxiv.org/pdf/2404.05311v1)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: We study the unique, less-well understood problem of generating sparse adversarial samples simply by observing the score-based replies to model queries. Sparse attacks aim to discover a minimum number-the l0 bounded-perturbations to model inputs to craft adversarial examples and misguide model decisions. But, in contrast to query-based dense attack counterparts against black-box models, constructing sparse adversarial perturbations, even when models serve confidence score information to queries in a score-based setting, is non-trivial. Because, such an attack leads to i) an NP-hard problem; and ii) a non-differentiable search space. We develop the BruSLeAttack-a new, faster (more query-efficient) Bayesian algorithm for the problem. We conduct extensive attack evaluations including an attack demonstration against a Machine Learning as a Service (MLaaS) offering exemplified by Google Cloud Vision and robustness testing of adversarial training regimes and a recent defense against black-box attacks. The proposed attack scales to achieve state-of-the-art attack success rates and query efficiency on standard computer vision tasks such as ImageNet across different model architectures. Our artefacts and DIY attack samples are available on GitHub. Importantly, our work facilitates faster evaluation of model vulnerabilities and raises our vigilance on the safety, security and reliability of deployed systems.



## **6. Out-of-Distribution Data: An Acquaintance of Adversarial Examples -- A Survey**

cs.LG

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05219v1) [paper-pdf](http://arxiv.org/pdf/2404.05219v1)

**Authors**: Naveen Karunanayake, Ravin Gunawardena, Suranga Seneviratne, Sanjay Chawla

**Abstract**: Deep neural networks (DNNs) deployed in real-world applications can encounter out-of-distribution (OOD) data and adversarial examples. These represent distinct forms of distributional shifts that can significantly impact DNNs' reliability and robustness. Traditionally, research has addressed OOD detection and adversarial robustness as separate challenges. This survey focuses on the intersection of these two areas, examining how the research community has investigated them together. Consequently, we identify two key research directions: robust OOD detection and unified robustness. Robust OOD detection aims to differentiate between in-distribution (ID) data and OOD data, even when they are adversarially manipulated to deceive the OOD detector. Unified robustness seeks a single approach to make DNNs robust against both adversarial attacks and OOD inputs. Accordingly, first, we establish a taxonomy based on the concept of distributional shifts. This framework clarifies how robust OOD detection and unified robustness relate to other research areas addressing distributional shifts, such as OOD detection, open set recognition, and anomaly detection. Subsequently, we review existing work on robust OOD detection and unified robustness. Finally, we highlight the limitations of the existing work and propose promising research directions that explore adversarial and OOD inputs within a unified framework.



## **7. Semantic Stealth: Adversarial Text Attacks on NLP Using Several Methods**

cs.CL

This report pertains to the Capstone Project done by Group 2 of the  Fall batch of 2023 students at Praxis Tech School, Kolkata, India. The  reports consists of 28 pages and it includes 10 tables. This is the preprint  which will be submitted to IEEE CONIT 2024 for review

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05159v1) [paper-pdf](http://arxiv.org/pdf/2404.05159v1)

**Authors**: Roopkatha Dey, Aivy Debnath, Sayak Kumar Dutta, Kaustav Ghosh, Arijit Mitra, Arghya Roy Chowdhury, Jaydip Sen

**Abstract**: In various real-world applications such as machine translation, sentiment analysis, and question answering, a pivotal role is played by NLP models, facilitating efficient communication and decision-making processes in domains ranging from healthcare to finance. However, a significant challenge is posed to the robustness of these natural language processing models by text adversarial attacks. These attacks involve the deliberate manipulation of input text to mislead the predictions of the model while maintaining human interpretability. Despite the remarkable performance achieved by state-of-the-art models like BERT in various natural language processing tasks, they are found to remain vulnerable to adversarial perturbations in the input text. In addressing the vulnerability of text classifiers to adversarial attacks, three distinct attack mechanisms are explored in this paper using the victim model BERT: BERT-on-BERT attack, PWWS attack, and Fraud Bargain's Attack (FBA). Leveraging the IMDB, AG News, and SST2 datasets, a thorough comparative analysis is conducted to assess the effectiveness of these attacks on the BERT classifier model. It is revealed by the analysis that PWWS emerges as the most potent adversary, consistently outperforming other methods across multiple evaluation scenarios, thereby emphasizing its efficacy in generating adversarial examples for text classification. Through comprehensive experimentation, the performance of these attacks is assessed and the findings indicate that the PWWS attack outperforms others, demonstrating lower runtime, higher accuracy, and favorable semantic similarity scores. The key insight of this paper lies in the assessment of the relative performances of three prevalent state-of-the-art attack mechanisms.



## **8. Enabling Privacy-Preserving Cyber Threat Detection with Federated Learning**

cs.CR

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05130v1) [paper-pdf](http://arxiv.org/pdf/2404.05130v1)

**Authors**: Yu Bi, Yekai Li, Xuan Feng, Xianghang Mi

**Abstract**: Despite achieving good performance and wide adoption, machine learning based security detection models (e.g., malware classifiers) are subject to concept drift and evasive evolution of attackers, which renders up-to-date threat data as a necessity. However, due to enforcement of various privacy protection regulations (e.g., GDPR), it is becoming increasingly challenging or even prohibitive for security vendors to collect individual-relevant and privacy-sensitive threat datasets, e.g., SMS spam/non-spam messages from mobile devices. To address such obstacles, this study systematically profiles the (in)feasibility of federated learning for privacy-preserving cyber threat detection in terms of effectiveness, byzantine resilience, and efficiency. This is made possible by the build-up of multiple threat datasets and threat detection models, and more importantly, the design of realistic and security-specific experiments.   We evaluate FL on two representative threat detection tasks, namely SMS spam detection and Android malware detection. It shows that FL-trained detection models can achieve a performance that is comparable to centrally trained counterparts. Also, most non-IID data distributions have either minor or negligible impact on the model performance, while a label-based non-IID distribution of a high extent can incur non-negligible fluctuation and delay in FL training. Then, under a realistic threat model, FL turns out to be adversary-resistant to attacks of both data poisoning and model poisoning. Particularly, the attacking impact of a practical data poisoning attack is no more than 0.14\% loss in model accuracy. Regarding FL efficiency, a bootstrapping strategy turns out to be effective to mitigate the training delay as observed in label-based non-IID scenarios.



## **9. Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations**

cs.LG

29 pages, 4 figures

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2402.05713v3) [paper-pdf](http://arxiv.org/pdf/2402.05713v3)

**Authors**: Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh

**Abstract**: The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce undetectable underdiagnosis bias in DL models. Our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups show that adversarial bias attacks demonstrate high-selectivity for bias in the targeted group by degrading group model performance without impacting overall model performance. Furthermore, our results indicate that adversarial bias attacks result in biased DL models that propagate prediction bias even when evaluated with external datasets.



## **10. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2402.08656v3) [paper-pdf](http://arxiv.org/pdf/2402.08656v3)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.



## **11. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2311.08268v4) [paper-pdf](http://arxiv.org/pdf/2311.08268v4)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.



## **12. Provable Robustness Against a Union of $\ell_0$ Adversarial Attacks**

cs.LG

Accepted at AAAI 2024 -- Extended version including the supplementary  material

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2302.11628v4) [paper-pdf](http://arxiv.org/pdf/2302.11628v4)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Sparse or $\ell_0$ adversarial attacks arbitrarily perturb an unknown subset of the features. $\ell_0$ robustness analysis is particularly well-suited for heterogeneous (tabular) data where features have different types or scales. State-of-the-art $\ell_0$ certified defenses are based on randomized smoothing and apply to evasion attacks only. This paper proposes feature partition aggregation (FPA) -- a certified defense against the union of $\ell_0$ evasion, backdoor, and poisoning attacks. FPA generates its stronger robustness guarantees via an ensemble whose submodels are trained on disjoint feature sets. Compared to state-of-the-art $\ell_0$ defenses, FPA is up to 3,000${\times}$ faster and provides larger median robustness guarantees (e.g., median certificates of 13 pixels over 10 for CIFAR10, 12 pixels over 10 for MNIST, 4 features over 1 for Weather, and 3 features over 1 for Ames), meaning FPA provides the additional dimensions of robustness essentially for free.



## **13. Data Poisoning Attacks on Off-Policy Policy Evaluation Methods**

cs.LG

Accepted at UAI 2022

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04714v1) [paper-pdf](http://arxiv.org/pdf/2404.04714v1)

**Authors**: Elita Lobo, Harvineet Singh, Marek Petrik, Cynthia Rudin, Himabindu Lakkaraju

**Abstract**: Off-policy Evaluation (OPE) methods are a crucial tool for evaluating policies in high-stakes domains such as healthcare, where exploration is often infeasible, unethical, or expensive. However, the extent to which such methods can be trusted under adversarial threats to data quality is largely unexplored. In this work, we make the first attempt at investigating the sensitivity of OPE methods to marginal adversarial perturbations to the data. We design a generic data poisoning attack framework leveraging influence functions from robust statistics to carefully construct perturbations that maximize error in the policy value estimates. We carry out extensive experimentation with multiple healthcare and control datasets. Our results demonstrate that many existing OPE methods are highly prone to generating value estimates with large errors when subject to data poisoning attacks, even for small adversarial perturbations. These findings question the reliability of policy values derived using OPE methods and motivate the need for developing OPE methods that are statistically robust to train-time data poisoning attacks.



## **14. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

cs.CL

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2310.00322v4) [paper-pdf](http://arxiv.org/pdf/2310.00322v4)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.



## **15. CANEDERLI: On The Impact of Adversarial Training and Transferability on CAN Intrusion Detection Systems**

cs.CR

Accepted at WiseML 2024

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04648v1) [paper-pdf](http://arxiv.org/pdf/2404.04648v1)

**Authors**: Francesco Marchiori, Mauro Conti

**Abstract**: The growing integration of vehicles with external networks has led to a surge in attacks targeting their Controller Area Network (CAN) internal bus. As a countermeasure, various Intrusion Detection Systems (IDSs) have been suggested in the literature to prevent and mitigate these threats. With the increasing volume of data facilitated by the integration of Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication networks, most of these systems rely on data-driven approaches such as Machine Learning (ML) and Deep Learning (DL) models. However, these systems are susceptible to adversarial evasion attacks. While many researchers have explored this vulnerability, their studies often involve unrealistic assumptions, lack consideration for a realistic threat model, and fail to provide effective solutions.   In this paper, we present CANEDERLI (CAN Evasion Detection ResiLIence), a novel framework for securing CAN-based IDSs. Our system considers a realistic threat model and addresses the impact of adversarial attacks on DL-based detection systems. Our findings highlight strong transferability properties among diverse attack methodologies by considering multiple state-of-the-art attacks and model architectures. We analyze the impact of adversarial training in addressing this threat and propose an adaptive online adversarial training technique outclassing traditional fine-tuning methodologies with F1 scores up to 0.941. By making our framework publicly available, we aid practitioners and researchers in assessing the resilience of IDSs to a varied adversarial landscape.



## **16. Dynamic Graph Information Bottleneck**

cs.LG

Accepted by the research tracks of The Web Conference 2024 (WWW 2024)

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2402.06716v3) [paper-pdf](http://arxiv.org/pdf/2402.06716v3)

**Authors**: Haonan Yuan, Qingyun Sun, Xingcheng Fu, Cheng Ji, Jianxin Li

**Abstract**: Dynamic Graphs widely exist in the real world, which carry complicated spatial and temporal feature patterns, challenging their representation learning. Dynamic Graph Neural Networks (DGNNs) have shown impressive predictive abilities by exploiting the intrinsic dynamics. However, DGNNs exhibit limited robustness, prone to adversarial attacks. This paper presents the novel Dynamic Graph Information Bottleneck (DGIB) framework to learn robust and discriminative representations. Leveraged by the Information Bottleneck (IB) principle, we first propose the expected optimal representations should satisfy the Minimal-Sufficient-Consensual (MSC) Condition. To compress redundant as well as conserve meritorious information into latent representation, DGIB iteratively directs and refines the structural and feature information flow passing through graph snapshots. To meet the MSC Condition, we decompose the overall IB objectives into DGIB$_{MS}$ and DGIB$_C$, in which the DGIB$_{MS}$ channel aims to learn the minimal and sufficient representations, with the DGIB$_{MS}$ channel guarantees the predictive consensus. Extensive experiments on real-world and synthetic dynamic graph datasets demonstrate the superior robustness of DGIB against adversarial attacks compared with state-of-the-art baselines in the link prediction task. To the best of our knowledge, DGIB is the first work to learn robust representations of dynamic graphs grounded in the information-theoretic IB principle.



## **17. Recovery from Adversarial Attacks in Cyber-physical Systems: Shallow, Deep and Exploratory Works**

eess.SY

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04472v1) [paper-pdf](http://arxiv.org/pdf/2404.04472v1)

**Authors**: Pengyuan Lu, Lin Zhang, Mengyu Liu, Kaustubh Sridhar, Fanxin Kong, Oleg Sokolsky, Insup Lee

**Abstract**: Cyber-physical systems (CPS) have experienced rapid growth in recent decades. However, like any other computer-based systems, malicious attacks evolve mutually, driving CPS to undesirable physical states and potentially causing catastrophes. Although the current state-of-the-art is well aware of this issue, the majority of researchers have not focused on CPS recovery, the procedure we defined as restoring a CPS's physical state back to a target condition under adversarial attacks. To call for attention on CPS recovery and identify existing efforts, we have surveyed a total of 30 relevant papers. We identify a major partition of the proposed recovery strategies: shallow recovery vs. deep recovery, where the former does not use a dedicated recovery controller while the latter does. Additionally, we surveyed exploratory research on topics that facilitate recovery. From these publications, we discuss the current state-of-the-art of CPS recovery, with respect to applications, attack type, attack surfaces and system dynamics. Then, we identify untouched sub-domains in this field and suggest possible future directions for researchers.



## **18. Increased LLM Vulnerabilities from Fine-tuning and Quantization**

cs.CR

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04392v1) [paper-pdf](http://arxiv.org/pdf/2404.04392v1)

**Authors**: Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Large Language Models (LLMs) have become very popular and have found use cases in many domains, such as chatbots, auto-task completion agents, and much more. However, LLMs are vulnerable to different types of attacks, such as jailbreaking, prompt injection attacks, and privacy leakage attacks. Foundational LLMs undergo adversarial and alignment training to learn not to generate malicious and toxic content. For specialized use cases, these foundational LLMs are subjected to fine-tuning or quantization for better performance and efficiency. We examine the impact of downstream tasks such as fine-tuning and quantization on LLM vulnerability. We test foundation models like Mistral, Llama, MosaicML, and their fine-tuned versions. Our research shows that fine-tuning and quantization reduces jailbreak resistance significantly, leading to increased LLM vulnerabilities. Finally, we demonstrate the utility of external guardrails in reducing LLM vulnerabilities.



## **19. Compositional Estimation of Lipschitz Constants for Deep Neural Networks**

cs.LG

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04375v1) [paper-pdf](http://arxiv.org/pdf/2404.04375v1)

**Authors**: Yuezhu Xu, S. Sivaranjani

**Abstract**: The Lipschitz constant plays a crucial role in certifying the robustness of neural networks to input perturbations and adversarial attacks, as well as the stability and safety of systems with neural network controllers. Therefore, estimation of tight bounds on the Lipschitz constant of neural networks is a well-studied topic. However, typical approaches involve solving a large matrix verification problem, the computational cost of which grows significantly for deeper networks. In this letter, we provide a compositional approach to estimate Lipschitz constants for deep feedforward neural networks by obtaining an exact decomposition of the large matrix verification problem into smaller sub-problems. We further obtain a closed-form solution that applies to most common neural network activation functions, which will enable rapid robustness and stability certificates for neural networks deployed in online control settings. Finally, we demonstrate through numerical experiments that our approach provides a steep reduction in computation time while yielding Lipschitz bounds that are very close to those achieved by state-of-the-art approaches.



## **20. Dissecting Distribution Inference**

cs.LG

Accepted at SaTML 2023 (updated Yifu's email address)

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2212.07591v2) [paper-pdf](http://arxiv.org/pdf/2212.07591v2)

**Authors**: Anshuman Suri, Yifu Lu, Yanjin Chen, David Evans

**Abstract**: A distribution inference attack aims to infer statistical properties of data used to train machine learning models. These attacks are sometimes surprisingly potent, but the factors that impact distribution inference risk are not well understood and demonstrated attacks often rely on strong and unrealistic assumptions such as full knowledge of training environments even in supposedly black-box threat scenarios. To improve understanding of distribution inference risks, we develop a new black-box attack that even outperforms the best known white-box attack in most settings. Using this new attack, we evaluate distribution inference risk while relaxing a variety of assumptions about the adversary's knowledge under black-box access, like known model architectures and label-only access. Finally, we evaluate the effectiveness of previously proposed defenses and introduce new defenses. We find that although noise-based defenses appear to be ineffective, a simple re-sampling defense can be highly effective. Code is available at https://github.com/iamgroot42/dissecting_distribution_inference



## **21. Evaluating Adversarial Robustness: A Comparison Of FGSM, Carlini-Wagner Attacks, And The Role of Distillation as Defense Mechanism**

cs.CR

This report pertains to the Capstone Project done by Group 1 of the  Fall batch of 2023 students at Praxis Tech School, Kolkata, India. The  reports consists of 35 pages and it includes 15 figures and 10 tables. This  is the preprint which will be submitted to to an IEEE international  conference for review

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04245v1) [paper-pdf](http://arxiv.org/pdf/2404.04245v1)

**Authors**: Trilokesh Ranjan Sarkar, Nilanjan Das, Pralay Sankar Maitra, Bijoy Some, Ritwik Saha, Orijita Adhikary, Bishal Bose, Jaydip Sen

**Abstract**: This technical report delves into an in-depth exploration of adversarial attacks specifically targeted at Deep Neural Networks (DNNs) utilized for image classification. The study also investigates defense mechanisms aimed at bolstering the robustness of machine learning models. The research focuses on comprehending the ramifications of two prominent attack methodologies: the Fast Gradient Sign Method (FGSM) and the Carlini-Wagner (CW) approach. These attacks are examined concerning three pre-trained image classifiers: Resnext50_32x4d, DenseNet-201, and VGG-19, utilizing the Tiny-ImageNet dataset. Furthermore, the study proposes the robustness of defensive distillation as a defense mechanism to counter FGSM and CW attacks. This defense mechanism is evaluated using the CIFAR-10 dataset, where CNN models, specifically resnet101 and Resnext50_32x4d, serve as the teacher and student models, respectively. The proposed defensive distillation model exhibits effectiveness in thwarting attacks such as FGSM. However, it is noted to remain susceptible to more sophisticated techniques like the CW attack. The document presents a meticulous validation of the proposed scheme. It provides detailed and comprehensive results, elucidating the efficacy and limitations of the defense mechanisms employed. Through rigorous experimentation and analysis, the study offers insights into the dynamics of adversarial attacks on DNNs, as well as the effectiveness of defensive strategies in mitigating their impact.



## **22. On Inherent Adversarial Robustness of Active Vision Systems**

cs.CV

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.00185v2) [paper-pdf](http://arxiv.org/pdf/2404.00185v2)

**Authors**: Amitangshu Mukherjee, Timur Ibrayev, Kaushik Roy

**Abstract**: Current Deep Neural Networks are vulnerable to adversarial examples, which alter their predictions by adding carefully crafted noise. Since human eyes are robust to such inputs, it is possible that the vulnerability stems from the standard way of processing inputs in one shot by processing every pixel with the same importance. In contrast, neuroscience suggests that the human vision system can differentiate salient features by (1) switching between multiple fixation points (saccades) and (2) processing the surrounding with a non-uniform external resolution (foveation). In this work, we advocate that the integration of such active vision mechanisms into current deep learning systems can offer robustness benefits. Specifically, we empirically demonstrate the inherent robustness of two active vision methods - GFNet and FALcon - under a black box threat model. By learning and inferencing based on downsampled glimpses obtained from multiple distinct fixation points within an input, we show that these active methods achieve (2-3) times greater robustness compared to a standard passive convolutional network under state-of-the-art adversarial attacks. More importantly, we provide illustrative and interpretable visualization analysis that demonstrates how performing inference from distinct fixation points makes active vision methods less vulnerable to malicious inputs.



## **23. Reliable Feature Selection for Adversarially Robust Cyber-Attack Detection**

cs.CR

24 pages, 17 tables, Annals of Telecommunications journal. arXiv  admin note: substantial text overlap with arXiv:2402.16912

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04188v1) [paper-pdf](http://arxiv.org/pdf/2404.04188v1)

**Authors**: João Vitorino, Miguel Silva, Eva Maia, Isabel Praça

**Abstract**: The growing cybersecurity threats make it essential to use high-quality data to train Machine Learning (ML) models for network traffic analysis, without noisy or missing data. By selecting the most relevant features for cyber-attack detection, it is possible to improve both the robustness and computational efficiency of the models used in a cybersecurity system. This work presents a feature selection and consensus process that combines multiple methods and applies them to several network datasets. Two different feature sets were selected and were used to train multiple ML models with regular and adversarial training. Finally, an adversarial evasion robustness benchmark was performed to analyze the reliability of the different feature sets and their impact on the susceptibility of the models to adversarial examples. By using an improved dataset with more data diversity, selecting the best time-related features and a more specific feature set, and performing adversarial training, the ML models were able to achieve a better adversarially robust generalization. The robustness of the models was significantly improved without their generalization to regular traffic flows being affected, without increases of false alarms, and without requiring too many computational resources, which enables a reliable detection of suspicious activity and perturbed traffic flows in enterprise computer networks.



## **24. Foundations of Cyber Resilience: The Confluence of Game, Control, and Learning Theories**

eess.SY

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.01205v2) [paper-pdf](http://arxiv.org/pdf/2404.01205v2)

**Authors**: Quanyan Zhu

**Abstract**: Cyber resilience is a complementary concept to cybersecurity, focusing on the preparation, response, and recovery from cyber threats that are challenging to prevent. Organizations increasingly face such threats in an evolving cyber threat landscape. Understanding and establishing foundations for cyber resilience provide a quantitative and systematic approach to cyber risk assessment, mitigation policy evaluation, and risk-informed defense design. A systems-scientific view toward cyber risks provides holistic and system-level solutions. This chapter starts with a systemic view toward cyber risks and presents the confluence of game theory, control theory, and learning theories, which are three major pillars for the design of cyber resilience mechanisms to counteract increasingly sophisticated and evolving threats in our networks and organizations. Game and control theoretic methods provide a set of modeling frameworks to capture the strategic and dynamic interactions between defenders and attackers. Control and learning frameworks together provide a feedback-driven mechanism that enables autonomous and adaptive responses to threats. Game and learning frameworks offer a data-driven approach to proactively reason about adversarial behaviors and resilient strategies. The confluence of the three lays the theoretical foundations for the analysis and design of cyber resilience. This chapter presents various theoretical paradigms, including dynamic asymmetric games, moving horizon control, conjectural learning, and meta-learning, as recent advances at the intersection. This chapter concludes with future directions and discussions of the role of neurosymbolic learning and the synergy between foundation models and game models in cyber resilience.



## **25. Less is More: Understanding Word-level Textual Adversarial Attack via n-gram Frequency Descend**

cs.CL

To be published in: 2024 IEEE Conference on Artificial Intelligence  (CAI 2024)

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2302.02568v3) [paper-pdf](http://arxiv.org/pdf/2302.02568v3)

**Authors**: Ning Lu, Shengcai Liu, Zhirui Zhang, Qi Wang, Haifeng Liu, Ke Tang

**Abstract**: Word-level textual adversarial attacks have demonstrated notable efficacy in misleading Natural Language Processing (NLP) models. Despite their success, the underlying reasons for their effectiveness and the fundamental characteristics of adversarial examples (AEs) remain obscure. This work aims to interpret word-level attacks by examining their $n$-gram frequency patterns. Our comprehensive experiments reveal that in approximately 90\% of cases, word-level attacks lead to the generation of examples where the frequency of $n$-grams decreases, a tendency we term as the $n$-gram Frequency Descend ($n$-FD). This finding suggests a straightforward strategy to enhance model robustness: training models using examples with $n$-FD. To examine the feasibility of this strategy, we employed the $n$-gram frequency information, as an alternative to conventional loss gradients, to generate perturbed examples in adversarial training. The experiment results indicate that the frequency-based approach performs comparably with the gradient-based approach in improving model robustness. Our research offers a novel and more intuitive perspective for understanding word-level textual adversarial attacks and proposes a new direction to improve model robustness.



## **26. Beyond the Bridge: Contention-Based Covert and Side Channel Attacks on Multi-GPU Interconnect**

cs.CR

Accepted to SEED 2024

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.03877v1) [paper-pdf](http://arxiv.org/pdf/2404.03877v1)

**Authors**: Yicheng Zhang, Ravan Nazaraliyev, Sankha Baran Dutta, Nael Abu-Ghazaleh, Andres Marquez, Kevin Barker

**Abstract**: High-speed interconnects, such as NVLink, are integral to modern multi-GPU systems, acting as a vital link between CPUs and GPUs. This study highlights the vulnerability of multi-GPU systems to covert and side channel attacks due to congestion on interconnects. An adversary can infer private information about a victim's activities by monitoring NVLink congestion without needing special permissions. Leveraging this insight, we develop a covert channel attack across two GPUs with a bandwidth of 45.5 kbps and a low error rate, and introduce a side channel attack enabling attackers to fingerprint applications through the shared NVLink interconnect.



## **27. An Anomaly Behavior Analysis Framework for Securing Autonomous Vehicle Perception**

cs.RO

20th ACS/IEEE International Conference on Computer Systems and  Applications (IEEE AICCSA 2023)

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2310.05041v3) [paper-pdf](http://arxiv.org/pdf/2310.05041v3)

**Authors**: Murad Mehrab Abrar, Salim Hariri

**Abstract**: As a rapidly growing cyber-physical platform, Autonomous Vehicles (AVs) are encountering more security challenges as their capabilities continue to expand. In recent years, adversaries are actively targeting the perception sensors of autonomous vehicles with sophisticated attacks that are not easily detected by the vehicles' control systems. This work proposes an Anomaly Behavior Analysis approach to detect a perception sensor attack against an autonomous vehicle. The framework relies on temporal features extracted from a physics-based autonomous vehicle behavior model to capture the normal behavior of vehicular perception in autonomous driving. By employing a combination of model-based techniques and machine learning algorithms, the proposed framework distinguishes between normal and abnormal vehicular perception behavior. To demonstrate the application of the framework in practice, we performed a depth camera attack experiment on an autonomous vehicle testbed and generated an extensive dataset. We validated the effectiveness of the proposed framework using this real-world data and released the dataset for public access. To our knowledge, this dataset is the first of its kind and will serve as a valuable resource for the research community in evaluating their intrusion detection techniques effectively.



## **28. ProLoc: Robust Location Proofs in Hindsight**

cs.CR

14 pages, 5 figures

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.04297v1) [paper-pdf](http://arxiv.org/pdf/2404.04297v1)

**Authors**: Roberta De Viti, Pierfrancesco Ingo, Isaac Sheff, Peter Druschel, Deepak Garg

**Abstract**: Many online services rely on self-reported locations of user devices like smartphones. To mitigate harm from falsified self-reported locations, the literature has proposed location proof services (LPSs), which provide proof of a device's location by corroborating its self-reported location using short-range radio contacts with either trusted infrastructure or nearby devices that also report their locations. This paper presents ProLoc, a new LPS that extends prior work in two ways. First, ProLoc relaxes prior work's proofs that a device was at a given location to proofs that a device was within distance "d" of a given location. We argue that these weaker proofs, which we call "region proofs", are important because (i) region proofs can be constructed with few requirements on device reporting behavior as opposed to precise location proofs, and (ii) a quantitative bound on a device's distance from a known epicenter is useful for many applications. For example, in the context of citizen reporting near an unexpected event (earthquake, violent protest, etc.), knowing the verified distances of the reporting devices from the event's epicenter would be valuable for ranking the reports by relevance or flagging fake reports. Second, ProLoc includes a novel mechanism to prevent collusion attacks where a set of attacker-controlled devices corroborate each others' false locations. Ours is the first mechanism that does not need additional infrastructure to handle attacks with made-up devices, which an attacker can create in any number at any location without any cost. For this, we rely on a variant of TrustRank applied to the self-reported trajectories and encounters of devices. Our goal is to prevent retroactive attacks where the adversary cannot predict ahead of time which fake location it will want to report, which is the case for the reporting of unexpected events.



## **29. Knowledge Distillation-Based Model Extraction Attack using Private Counterfactual Explanations**

cs.LG

15 pages

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03348v1) [paper-pdf](http://arxiv.org/pdf/2404.03348v1)

**Authors**: Fatima Ezzeddine, Omran Ayoub, Silvia Giordano

**Abstract**: In recent years, there has been a notable increase in the deployment of machine learning (ML) models as services (MLaaS) across diverse production software applications. In parallel, explainable AI (XAI) continues to evolve, addressing the necessity for transparency and trustworthiness in ML models. XAI techniques aim to enhance the transparency of ML models by providing insights, in terms of the model's explanations, into their decision-making process. Simultaneously, some MLaaS platforms now offer explanations alongside the ML prediction outputs. This setup has elevated concerns regarding vulnerabilities in MLaaS, particularly in relation to privacy leakage attacks such as model extraction attacks (MEA). This is due to the fact that explanations can unveil insights about the inner workings of the model which could be exploited by malicious users. In this work, we focus on investigating how model explanations, particularly Generative adversarial networks (GANs)-based counterfactual explanations (CFs), can be exploited for performing MEA within the MLaaS platform. We also delve into assessing the effectiveness of incorporating differential privacy (DP) as a mitigation strategy. To this end, we first propose a novel MEA methodology based on Knowledge Distillation (KD) to enhance the efficiency of extracting a substitute model of a target model exploiting CFs. Then, we advise an approach for training CF generators incorporating DP to generate private CFs. We conduct thorough experimental evaluations on real-world datasets and demonstrate that our proposed KD-based MEA can yield a high-fidelity substitute model with reduced queries with respect to baseline approaches. Furthermore, our findings reveal that the inclusion of a privacy layer impacts the performance of the explainer, the quality of CFs, and results in a reduction in the MEA performance.



## **30. Meta Invariance Defense Towards Generalizable Robustness to Unknown Adversarial Attacks**

cs.CV

Accepted by IEEE TPAMI in 2024

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03340v1) [paper-pdf](http://arxiv.org/pdf/2404.03340v1)

**Authors**: Lei Zhang, Yuhang Zhou, Yi Yang, Xinbo Gao

**Abstract**: Despite providing high-performance solutions for computer vision tasks, the deep neural network (DNN) model has been proved to be extremely vulnerable to adversarial attacks. Current defense mainly focuses on the known attacks, but the adversarial robustness to the unknown attacks is seriously overlooked. Besides, commonly used adaptive learning and fine-tuning technique is unsuitable for adversarial defense since it is essentially a zero-shot problem when deployed. Thus, to tackle this challenge, we propose an attack-agnostic defense method named Meta Invariance Defense (MID). Specifically, various combinations of adversarial attacks are randomly sampled from a manually constructed Attacker Pool to constitute different defense tasks against unknown attacks, in which a student encoder is supervised by multi-consistency distillation to learn the attack-invariant features via a meta principle. The proposed MID has two merits: 1) Full distillation from pixel-, feature- and prediction-level between benign and adversarial samples facilitates the discovery of attack-invariance. 2) The model simultaneously achieves robustness to the imperceptible adversarial perturbations in high-level image classification and attack-suppression in low-level robust image regeneration. Theoretical and empirical studies on numerous benchmarks such as ImageNet verify the generalizable robustness and superiority of MID under various attacks.



## **31. Learn What You Want to Unlearn: Unlearning Inversion Attacks against Machine Unlearning**

cs.CR

To Appear in the 45th IEEE Symposium on Security and Privacy, May  20-23, 2024

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03233v1) [paper-pdf](http://arxiv.org/pdf/2404.03233v1)

**Authors**: Hongsheng Hu, Shuo Wang, Tian Dong, Minhui Xue

**Abstract**: Machine unlearning has become a promising solution for fulfilling the "right to be forgotten", under which individuals can request the deletion of their data from machine learning models. However, existing studies of machine unlearning mainly focus on the efficacy and efficiency of unlearning methods, while neglecting the investigation of the privacy vulnerability during the unlearning process. With two versions of a model available to an adversary, that is, the original model and the unlearned model, machine unlearning opens up a new attack surface. In this paper, we conduct the first investigation to understand the extent to which machine unlearning can leak the confidential content of the unlearned data. Specifically, under the Machine Learning as a Service setting, we propose unlearning inversion attacks that can reveal the feature and label information of an unlearned sample by only accessing the original and unlearned model. The effectiveness of the proposed unlearning inversion attacks is evaluated through extensive experiments on benchmark datasets across various model architectures and on both exact and approximate representative unlearning approaches. The experimental results indicate that the proposed attack can reveal the sensitive information of the unlearned data. As such, we identify three possible defenses that help to mitigate the proposed attacks, while at the cost of reducing the utility of the unlearned model. The study in this paper uncovers an underexplored gap between machine unlearning and the privacy of unlearned data, highlighting the need for the careful design of mechanisms for implementing unlearning without leaking the information of the unlearned data.



## **32. FACTUAL: A Novel Framework for Contrastive Learning Based Robust SAR Image Classification**

cs.CV

2024 IEEE Radar Conference

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03225v1) [paper-pdf](http://arxiv.org/pdf/2404.03225v1)

**Authors**: Xu Wang, Tian Ye, Rajgopal Kannan, Viktor Prasanna

**Abstract**: Deep Learning (DL) Models for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR), while delivering improved performance, have been shown to be quite vulnerable to adversarial attacks. Existing works improve robustness by training models on adversarial samples. However, by focusing mostly on attacks that manipulate images randomly, they neglect the real-world feasibility of such attacks. In this paper, we propose FACTUAL, a novel Contrastive Learning framework for Adversarial Training and robust SAR classification. FACTUAL consists of two components: (1) Differing from existing works, a novel perturbation scheme that incorporates realistic physical adversarial attacks (such as OTSA) to build a supervised adversarial pre-training network. This network utilizes class labels for clustering clean and perturbed images together into a more informative feature space. (2) A linear classifier cascaded after the encoder to use the computed representations to predict the target labels. By pre-training and fine-tuning our model on both clean and adversarial samples, we show that our model achieves high prediction accuracy on both cases. Our model achieves 99.7% accuracy on clean samples, and 89.6% on perturbed samples, both outperforming previous state-of-the-art methods.



## **33. Robust Federated Learning for Wireless Networks: A Demonstration with Channel Estimation**

cs.LG

Submitted to IEEE GLOBECOM 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.03088v1) [paper-pdf](http://arxiv.org/pdf/2404.03088v1)

**Authors**: Zexin Fang, Bin Han, Hans D. Schotten

**Abstract**: Federated learning (FL) offers a privacy-preserving collaborative approach for training models in wireless networks, with channel estimation emerging as a promising application. Despite extensive studies on FL-empowered channel estimation, the security concerns associated with FL require meticulous attention. In a scenario where small base stations (SBSs) serve as local models trained on cached data, and a macro base station (MBS) functions as the global model setting, an attacker can exploit the vulnerability of FL, launching attacks with various adversarial attacks or deployment tactics. In this paper, we analyze such vulnerabilities, corresponding solutions were brought forth, and validated through simulation.



## **34. Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning**

cs.CR

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2306.05494v2) [paper-pdf](http://arxiv.org/pdf/2306.05494v2)

**Authors**: Mohamed el Shehaby, Ashraf Matrawy

**Abstract**: Machine Learning (ML) has become ubiquitous, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy compared to traditional models in processing and classifying large volumes of data. However, ML has been found to have several flaws, most importantly, adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the suitability of these attacks against ML-based network security entities, especially NIDS, due to the wide difference between different domains regarding the generation of adversarial attacks.   To further explore the practicality of adversarial attacks against ML-based NIDS in-depth, this paper presents three distinct contributions: identifying numerous practicality issues for evasion adversarial attacks on ML-NIDS using an attack tree threat model, introducing a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS, and investigating how the dynamicity of some real-world ML models affects adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effectiveness of adversarial attacks. While adversarial attacks can compromise ML-based NIDSs, our aim is to highlight the significant gap between research and real-world practicality in this domain, warranting attention.



## **35. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.03027v1) [paper-pdf](http://arxiv.org/pdf/2404.03027v1)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.



## **36. Adversarial Attacks and Dimensionality in Text Classifiers**

cs.LG

This paper is accepted for publication at EURASIP Journal on  Information Security in 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02660v1) [paper-pdf](http://arxiv.org/pdf/2404.02660v1)

**Authors**: Nandish Chattopadhyay, Atreya Goswami, Anupam Chattopadhyay

**Abstract**: Adversarial attacks on machine learning algorithms have been a key deterrent to the adoption of AI in many real-world use cases. They significantly undermine the ability of high-performance neural networks by forcing misclassifications. These attacks introduce minute and structured perturbations or alterations in the test samples, imperceptible to human annotators in general, but trained neural networks and other models are sensitive to it. Historically, adversarial attacks have been first identified and studied in the domain of image processing. In this paper, we study adversarial examples in the field of natural language processing, specifically text classification tasks. We investigate the reasons for adversarial vulnerability, particularly in relation to the inherent dimensionality of the model. Our key finding is that there is a very strong correlation between the embedding dimensionality of the adversarial samples and their effectiveness on models tuned with input samples with same embedding dimension. We utilize this sensitivity to design an adversarial defense mechanism. We use ensemble models of varying inherent dimensionality to thwart the attacks. This is tested on multiple datasets for its efficacy in providing robustness. We also study the problem of measuring adversarial perturbation using different distance metrics. For all of the aforementioned studies, we have run tests on multiple models with varying dimensionality and used a word-vector level adversarial attack to substantiate the findings.



## **37. Adversary-Augmented Simulation to evaluate fairness on HyperLedger Fabric**

cs.CR

10 pages, 8 figures

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2403.14342v2) [paper-pdf](http://arxiv.org/pdf/2403.14342v2)

**Authors**: Erwan Mahe, Rouwaida Abdallah, Sara Tucci-Piergiovanni, Pierre-Yves Piriou

**Abstract**: This paper presents a novel adversary model specifically tailored to distributed systems, aiming to assess the security of blockchain networks. Building upon concepts such as adversarial assumptions, goals, and capabilities, our proposed adversary model classifies and constrains the use of adversarial actions based on classical distributed system models, defined by both failure and communication models. The objective is to study the effects of these allowed actions on the properties of distributed protocols under various system models. A significant aspect of our research involves integrating this adversary model into the Multi-Agent eXperimenter (MAX) framework. This integration enables fine-grained simulations of adversarial attacks on blockchain networks. In this paper, we particularly study four distinct fairness properties on Hyperledger Fabric with the Byzantine Fault Tolerant Tendermint consensus algorithm being selected for its ordering service. We define novel attacks that combine adversarial actions on both protocols, with the aim of violating a specific client-fairness property. Simulations confirm our ability to violate this property and allow us to evaluate the impact of these attacks on several order-fairness properties that relate orders of transaction reception and delivery.



## **38. Unsegment Anything by Simulating Deformation**

cs.CV

CVPR 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02585v1) [paper-pdf](http://arxiv.org/pdf/2404.02585v1)

**Authors**: Jiahao Lu, Xingyi Yang, Xinchao Wang

**Abstract**: Foundation segmentation models, while powerful, pose a significant risk: they enable users to effortlessly extract any objects from any digital content with a single click, potentially leading to copyright infringement or malicious misuse. To mitigate this risk, we introduce a new task "Anything Unsegmentable" to grant any image "the right to be unsegmented". The ambitious pursuit of the task is to achieve highly transferable adversarial attacks against all prompt-based segmentation models, regardless of model parameterizations and prompts. We highlight the non-transferable and heterogeneous nature of prompt-specific adversarial noises. Our approach focuses on disrupting image encoder features to achieve prompt-agnostic attacks. Intriguingly, targeted feature attacks exhibit better transferability compared to untargeted ones, suggesting the optimal update direction aligns with the image manifold. Based on the observations, we design a novel attack named Unsegment Anything by Simulating Deformation (UAD). Our attack optimizes a differentiable deformation function to create a target deformed image, which alters structural information while preserving achievable feature distance by adversarial example. Extensive experiments verify the effectiveness of our approach, compromising a variety of promptable segmentation models with different architectures and prompt interfaces. We release the code at https://github.com/jiahaolu97/anything-unsegmentable.



## **39. A Unified Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability**

cs.CV

Membership Inference, Self-supervised learning

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02462v1) [paper-pdf](http://arxiv.org/pdf/2404.02462v1)

**Authors**: Jie Zhu, Jirong Zha, Ding Li, Leye Wang

**Abstract**: Self-supervised learning shows promise in harnessing extensive unlabeled data, but it also confronts significant privacy concerns, especially in vision. In this paper, we aim to perform membership inference on visual self-supervised models in a more realistic setting: self-supervised training method and details are unknown for an adversary when attacking as he usually faces a black-box system in practice. In this setting, considering that self-supervised model could be trained by completely different self-supervised paradigms, e.g., masked image modeling and contrastive learning, with complex training details, we propose a unified membership inference method called PartCrop. It is motivated by the shared part-aware capability among models and stronger part response on the training data. Specifically, PartCrop crops parts of objects in an image to query responses with the image in representation space. We conduct extensive attacks on self-supervised models with different training protocols and structures using three widely used image datasets. The results verify the effectiveness and generalization of PartCrop. Moreover, to defend against PartCrop, we evaluate two common approaches, i.e., early stop and differential privacy, and propose a tailored method called shrinking crop scale range. The defense experiments indicate that all of them are effective. Our code is available at https://github.com/JiePKU/PartCrop



## **40. Designing a Photonic Physically Unclonable Function Having Resilience to Machine Learning Attacks**

cs.CR

14 pages, 8 figures

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02440v1) [paper-pdf](http://arxiv.org/pdf/2404.02440v1)

**Authors**: Elena R. Henderson, Jessie M. Henderson, Hiva Shahoei, William V. Oxford, Eric C. Larson, Duncan L. MacFarlane, Mitchell A. Thornton

**Abstract**: Physically unclonable functions (PUFs) are designed to act as device 'fingerprints.' Given an input challenge, the PUF circuit should produce an unpredictable response for use in situations such as root-of-trust applications and other hardware-level cybersecurity applications. PUFs are typically subcircuits present within integrated circuits (ICs), and while conventional IC PUFs are well-understood, several implementations have proven vulnerable to malicious exploits, including those perpetrated by machine learning (ML)-based attacks. Such attacks can be difficult to prevent because they are often designed to work even when relatively few challenge-response pairs are known in advance. Hence the need for both more resilient PUF designs and analysis of ML-attack susceptibility. Previous work has developed a PUF for photonic integrated circuits (PICs). A PIC PUF not only produces unpredictable responses given manufacturing-introduced tolerances, but is also less prone to electromagnetic radiation eavesdropping attacks than a purely electronic IC PUF. In this work, we analyze the resilience of the proposed photonic PUF when subjected to ML-based attacks. Specifically, we describe a computational PUF model for producing the large datasets required for training ML attacks; we analyze the quality of the model; and we discuss the modeled PUF's susceptibility to ML-based attacks. We find that the modeled PUF generates distributions that resemble uniform white noise, explaining the exhibited resilience to neural-network-based attacks designed to exploit latent relationships between challenges and responses. Preliminary analysis suggests that the PUF exhibits similar resilience to generative adversarial networks, and continued development will show whether more-sophisticated ML approaches better compromise the PUF and -- if so -- how design modifications might improve resilience.



## **41. One Noise to Rule Them All: Multi-View Adversarial Attacks with Universal Perturbation**

cs.CV

6 pages, 4 figures, presented at ICAIA, Springer to publish under  Algorithms for Intelligent Systems

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02287v1) [paper-pdf](http://arxiv.org/pdf/2404.02287v1)

**Authors**: Mehmet Ergezer, Phat Duong, Christian Green, Tommy Nguyen, Abdurrahman Zeybey

**Abstract**: This paper presents a novel universal perturbation method for generating robust multi-view adversarial examples in 3D object recognition. Unlike conventional attacks limited to single views, our approach operates on multiple 2D images, offering a practical and scalable solution for enhancing model scalability and robustness. This generalizable method bridges the gap between 2D perturbations and 3D-like attack capabilities, making it suitable for real-world applications.   Existing adversarial attacks may become ineffective when images undergo transformations like changes in lighting, camera position, or natural deformations. We address this challenge by crafting a single universal noise perturbation applicable to various object views. Experiments on diverse rendered 3D objects demonstrate the effectiveness of our approach. The universal perturbation successfully identified a single adversarial noise for each given set of 3D object renders from multiple poses and viewpoints. Compared to single-view attacks, our universal attacks lower classification confidence across multiple viewing angles, especially at low noise levels. A sample implementation is made available at https://github.com/memoatwit/UniversalPerturbation.



## **42. Towards Robust 3D Pose Transfer with Adversarial Learning**

cs.CV

CVPR 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02242v1) [paper-pdf](http://arxiv.org/pdf/2404.02242v1)

**Authors**: Haoyu Chen, Hao Tang, Ehsan Adeli, Guoying Zhao

**Abstract**: 3D pose transfer that aims to transfer the desired pose to a target mesh is one of the most challenging 3D generation tasks. Previous attempts rely on well-defined parametric human models or skeletal joints as driving pose sources. However, to obtain those clean pose sources, cumbersome but necessary pre-processing pipelines are inevitable, hindering implementations of the real-time applications. This work is driven by the intuition that the robustness of the model can be enhanced by introducing adversarial samples into the training, leading to a more invulnerable model to the noisy inputs, which even can be further extended to directly handling the real-world data like raw point clouds/scans without intermediate processing. Furthermore, we propose a novel 3D pose Masked Autoencoder (3D-PoseMAE), a customized MAE that effectively learns 3D extrinsic presentations (i.e., pose). 3D-PoseMAE facilitates learning from the aspect of extrinsic attributes by simultaneously generating adversarial samples that perturb the model and learning the arbitrary raw noisy poses via a multi-scale masking strategy. Both qualitative and quantitative studies show that the transferred meshes given by our network result in much better quality. Besides, we demonstrate the strong generalizability of our method on various poses, different domains, and even raw scans. Experimental results also show meaningful insights that the intermediate adversarial samples generated in the training can successfully attack the existing pose transfer models.



## **43. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02151v1) [paper-pdf](http://arxiv.org/pdf/2404.02151v1)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). We provide the code, prompts, and logs of the attacks at https://github.com/tml-epfl/llm-adaptive-attacks.



## **44. READ: Improving Relation Extraction from an ADversarial Perspective**

cs.CL

Accepted by findings of NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02931v1) [paper-pdf](http://arxiv.org/pdf/2404.02931v1)

**Authors**: Dawei Li, William Hogan, Jingbo Shang

**Abstract**: Recent works in relation extraction (RE) have achieved promising benchmark accuracy; however, our adversarial attack experiments show that these works excessively rely on entities, making their generalization capability questionable. To address this issue, we propose an adversarial training method specifically designed for RE. Our approach introduces both sequence- and token-level perturbations to the sample and uses a separate perturbation vocabulary to improve the search for entity and context perturbations. Furthermore, we introduce a probabilistic strategy for leaving clean tokens in the context during adversarial training. This strategy enables a larger attack budget for entities and coaxes the model to leverage relational patterns embedded in the context. Extensive experiments show that compared to various adversarial training methods, our method significantly improves both the accuracy and robustness of the model. Additionally, experiments on different data availability settings highlight the effectiveness of our method in low-resource scenarios. We also perform in-depth analyses of our proposed method and provide further hints. We will release our code at https://github.com/David-Li0406/READ.



## **45. Red-Teaming Segment Anything Model**

cs.CV

CVPR 2024 - The 4th Workshop of Adversarial Machine Learning on  Computer Vision: Robustness of Foundation Models

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02067v1) [paper-pdf](http://arxiv.org/pdf/2404.02067v1)

**Authors**: Krzysztof Jankowski, Bartlomiej Sobieski, Mateusz Kwiatkowski, Jakub Szulc, Michal Janik, Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Foundation models have emerged as pivotal tools, tackling many complex tasks through pre-training on vast datasets and subsequent fine-tuning for specific applications. The Segment Anything Model is one of the first and most well-known foundation models for computer vision segmentation tasks. This work presents a multi-faceted red-teaming analysis that tests the Segment Anything Model against challenging tasks: (1) We analyze the impact of style transfer on segmentation masks, demonstrating that applying adverse weather conditions and raindrops to dashboard images of city roads significantly distorts generated masks. (2) We focus on assessing whether the model can be used for attacks on privacy, such as recognizing celebrities' faces, and show that the model possesses some undesired knowledge in this task. (3) Finally, we check how robust the model is to adversarial attacks on segmentation masks under text prompts. We not only show the effectiveness of popular white-box attacks and resistance to black-box attacks but also introduce a novel approach - Focused Iterative Gradient Attack (FIGA) that combines white-box approaches to construct an efficient attack resulting in a smaller number of modified pixels. All of our testing methods and analyses indicate a need for enhanced safety measures in foundation models for image segmentation.



## **46. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

cs.CL

NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2311.09447v2) [paper-pdf](http://arxiv.org/pdf/2311.09447v2)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an adversarial assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose advCoU, an extended Chain of Utterances-based (CoU) prompting strategy by incorporating carefully crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.



## **47. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16591v3) [paper-pdf](http://arxiv.org/pdf/2403.16591v3)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between LDP and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. The relationship between LDP and Maximum Bayesian Privacy (MBP) is first revealed, demonstrating that under uniform prior distribution, a mechanism satisfying $\xi$-LDP will satisfy $\xi$-MBP and conversely $\xi$-MBP also confers 2$\xi$-LDP. Our next theoretical contribution are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms. Our work not only lays the groundwork for future empirical exploration but also promises to facilitate the design of privacy-preserving algorithms, thereby fostering the development of trustworthy machine learning solutions.



## **48. PatchCURE: Improving Certifiable Robustness, Model Utility, and Computation Efficiency of Adversarial Patch Defenses**

cs.CV

USENIX Security 2024. (extended) technical report

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2310.13076v2) [paper-pdf](http://arxiv.org/pdf/2310.13076v2)

**Authors**: Chong Xiang, Tong Wu, Sihui Dai, Jonathan Petit, Suman Jana, Prateek Mittal

**Abstract**: State-of-the-art defenses against adversarial patch attacks can now achieve strong certifiable robustness with a marginal drop in model utility. However, this impressive performance typically comes at the cost of 10-100x more inference-time computation compared to undefended models -- the research community has witnessed an intense three-way trade-off between certifiable robustness, model utility, and computation efficiency. In this paper, we propose a defense framework named PatchCURE to approach this trade-off problem. PatchCURE provides sufficient "knobs" for tuning defense performance and allows us to build a family of defenses: the most robust PatchCURE instance can match the performance of any existing state-of-the-art defense (without efficiency considerations); the most efficient PatchCURE instance has similar inference efficiency as undefended models. Notably, PatchCURE achieves state-of-the-art robustness and utility performance across all different efficiency levels, e.g., 16-23% absolute clean accuracy and certified robust accuracy advantages over prior defenses when requiring computation efficiency to be close to undefended models. The family of PatchCURE defenses enables us to flexibly choose appropriate defenses to satisfy given computation and/or utility constraints in practice.



## **49. Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack**

cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01907v1) [paper-pdf](http://arxiv.org/pdf/2404.01907v1)

**Authors**: Ying Zhou, Ben He, Le Sun

**Abstract**: With the development of large language models (LLMs), detecting whether text is generated by a machine becomes increasingly challenging in the face of malicious use cases like the spread of false information, protection of intellectual property, and prevention of academic plagiarism. While well-trained text detectors have demonstrated promising performance on unseen test data, recent research suggests that these detectors have vulnerabilities when dealing with adversarial attacks such as paraphrasing. In this paper, we propose a framework for a broader class of adversarial attacks, designed to perform minor perturbations in machine-generated content to evade detection. We consider two attack settings: white-box and black-box, and employ adversarial learning in dynamic scenarios to assess the potential enhancement of the current detection model's robustness against such attacks. The empirical results reveal that the current detection models can be compromised in as little as 10 seconds, leading to the misclassification of machine-generated text as human-written content. Furthermore, we explore the prospect of improving the model's robustness over iterative adversarial learning. Although some improvements in model robustness are observed, practical applications still face significant challenges. These findings shed light on the future development of AI-text detectors, emphasizing the need for more accurate and robust detection methods.



## **50. Defense without Forgetting: Continual Adversarial Defense with Anisotropic & Isotropic Pseudo Replay**

cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01828v1) [paper-pdf](http://arxiv.org/pdf/2404.01828v1)

**Authors**: Yuhang Zhou, Zhongyun Hua

**Abstract**: Deep neural networks have demonstrated susceptibility to adversarial attacks. Adversarial defense techniques often focus on one-shot setting to maintain robustness against attack. However, new attacks can emerge in sequences in real-world deployment scenarios. As a result, it is crucial for a defense model to constantly adapt to new attacks, but the adaptation process can lead to catastrophic forgetting of previously defended against attacks. In this paper, we discuss for the first time the concept of continual adversarial defense under a sequence of attacks, and propose a lifelong defense baseline called Anisotropic \& Isotropic Replay (AIR), which offers three advantages: (1) Isotropic replay ensures model consistency in the neighborhood distribution of new data, indirectly aligning the output preference between old and new tasks. (2) Anisotropic replay enables the model to learn a compromise data manifold with fresh mixed semantics for further replay constraints and potential future attacks. (3) A straightforward regularizer mitigates the 'plasticity-stability' trade-off by aligning model output between new and old tasks. Experiment results demonstrate that AIR can approximate or even exceed the empirical performance upper bounds achieved by Joint Training.



