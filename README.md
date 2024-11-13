# Latest Adversarial Attack Papers
**update at 2024-11-13 17:30:03**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Can adversarial attacks by large language models be attributed?**

cs.AI

7 pages, 1 figure

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08003v1) [paper-pdf](http://arxiv.org/pdf/2411.08003v1)

**Authors**: Manuel Cebrian, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation-presents significant challenges that are likely to grow in importance. We investigate this attribution problem using formal language theory, specifically language identification in the limit as introduced by Gold and extended by Angluin. By modeling LLM outputs as formal languages, we analyze whether finite text samples can uniquely pinpoint the originating model. Our results show that due to the non-identifiability of certain language classes, under some mild assumptions about overlapping outputs from fine-tuned models it is theoretically impossible to attribute outputs to specific LLMs with certainty. This holds also when accounting for expressivity limitations of Transformer architectures. Even with direct model access or comprehensive monitoring, significant computational hurdles impede attribution efforts. These findings highlight an urgent need for proactive measures to mitigate risks posed by adversarial LLM use as their influence continues to expand.



## **2. IAE: Irony-based Adversarial Examples for Sentiment Analysis Systems**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07850v1) [paper-pdf](http://arxiv.org/pdf/2411.07850v1)

**Authors**: Xiaoyin Yi, Jiacheng Huang

**Abstract**: Adversarial examples, which are inputs deliberately perturbed with imperceptible changes to induce model errors, have raised serious concerns for the reliability and security of deep neural networks (DNNs). While adversarial attacks have been extensively studied in continuous data domains such as images, the discrete nature of text presents unique challenges. In this paper, we propose Irony-based Adversarial Examples (IAE), a method that transforms straightforward sentences into ironic ones to create adversarial text. This approach exploits the rhetorical device of irony, where the intended meaning is opposite to the literal interpretation, requiring a deeper understanding of context to detect. The IAE method is particularly challenging due to the need to accurately locate evaluation words, substitute them with appropriate collocations, and expand the text with suitable ironic elements while maintaining semantic coherence. Our research makes the following key contributions: (1) We introduce IAE, a strategy for generating textual adversarial examples using irony. This method does not rely on pre-existing irony corpora, making it a versatile tool for creating adversarial text in various NLP tasks. (2) We demonstrate that the performance of several state-of-the-art deep learning models on sentiment analysis tasks significantly deteriorates when subjected to IAE attacks. This finding underscores the susceptibility of current NLP systems to adversarial manipulation through irony. (3) We compare the impact of IAE on human judgment versus NLP systems, revealing that humans are less susceptible to the effects of irony in text.



## **3. Chain Association-based Attacking and Shielding Natural Language Processing Systems**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07843v1) [paper-pdf](http://arxiv.org/pdf/2411.07843v1)

**Authors**: Jiacheng Huang, Long Chen

**Abstract**: Association as a gift enables people do not have to mention something in completely straightforward words and allows others to understand what they intend to refer to. In this paper, we propose a chain association-based adversarial attack against natural language processing systems, utilizing the comprehension gap between humans and machines. We first generate a chain association graph for Chinese characters based on the association paradigm for building search space of potential adversarial examples. Then, we introduce an discrete particle swarm optimization algorithm to search for the optimal adversarial examples. We conduct comprehensive experiments and show that advanced natural language processing models and applications, including large language models, are vulnerable to our attack, while humans appear good at understanding the perturbed text. We also explore two methods, including adversarial training and associative graph-based recovery, to shield systems from chain association-based attack. Since a few examples that use some derogatory terms, this paper contains materials that may be offensive or upsetting to some people.



## **4. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2410.23091v3) [paper-pdf](http://arxiv.org/pdf/2410.23091v3)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark).



## **5. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

cs.CV

17 pages, 13 figures

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2404.19287v3) [paper-pdf](http://arxiv.org/pdf/2404.19287v3)

**Authors**: Wanqi Zhou, Shuanghao Bai, Danilo P. Mandic, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP exhibit exceptional generalization across diverse downstream tasks. While recent studies reveal their vulnerability to adversarial attacks, research to date has primarily focused on enhancing the robustness of image encoders against image-based attacks, with defenses against text-based and multimodal attacks remaining largely unexplored. To this end, this work presents the first comprehensive study on improving the adversarial robustness of VLMs against attacks targeting image, text, and multimodal inputs. This is achieved by proposing multimodal contrastive adversarial training (MMCoA). Such an approach strengthens the robustness of both image and text encoders by aligning the clean text embeddings with adversarial image embeddings, and adversarial text embeddings with clean image embeddings. The robustness of the proposed MMCoA is examined against existing defense methods over image, text, and multimodal attacks on the CLIP model. Extensive experiments on 15 datasets across two tasks reveal the characteristics of different adversarial defense methods under distinct distribution shifts and dataset complexities across the three attack types. This paves the way for a unified framework of adversarial robustness against different modality attacks, opening up new possibilities for securing VLMs against multimodal attacks. The code is available at https://github.com/ElleZWQ/MMCoA.git.



## **6. Data-Driven Graph Switching for Cyber-Resilient Control in Microgrids**

eess.SY

Accepted in IEEE Design Methodologies Conference (DMC) 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07686v1) [paper-pdf](http://arxiv.org/pdf/2411.07686v1)

**Authors**: Suman Rath, Subham Sahoo

**Abstract**: Distributed microgrids are conventionally dependent on communication networks to achieve secondary control objectives. This dependence makes them vulnerable to stealth data integrity attacks (DIAs) where adversaries may perform manipulations via infected transmitters and repeaters to jeopardize stability. This paper presents a physics-guided, supervised Artificial Neural Network (ANN)-based framework that identifies communication-level cyberattacks in microgrids by analyzing whether incoming measurements will cause abnormal behavior of the secondary control layer. If abnormalities are detected, an iteration through possible spanning tree graph topologies that can be used to fulfill secondary control objectives is done. Then, a communication network topology that would not create secondary control abnormalities is identified and enforced for maximum stability. By altering the communication graph topology, the framework eliminates the dependence of the secondary control layer on inputs from compromised cyber devices helping it achieve resilience without instability. Several case studies are provided showcasing the robustness of the framework against False Data Injections and repeater-level Man-in-the-Middle attacks. To understand practical feasibility, robustness is also verified against larger microgrid sizes and in the presence of varying noise levels. Our findings indicate that performance can be affected when attempting scalability in the presence of noise. However, the framework operates robustly in low-noise settings.



## **7. A Survey on Adversarial Machine Learning for Code Data: Realistic Threats, Countermeasures, and Interpretations**

cs.CR

Under a reviewing process since Sep. 3, 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07597v1) [paper-pdf](http://arxiv.org/pdf/2411.07597v1)

**Authors**: Yulong Yang, Haoran Fan, Chenhao Lin, Qian Li, Zhengyu Zhao, Chao Shen, Xiaohong Guan

**Abstract**: Code Language Models (CLMs) have achieved tremendous progress in source code understanding and generation, leading to a significant increase in research interests focused on applying CLMs to real-world software engineering tasks in recent years. However, in realistic scenarios, CLMs are exposed to potential malicious adversaries, bringing risks to the confidentiality, integrity, and availability of CLM systems. Despite these risks, a comprehensive analysis of the security vulnerabilities of CLMs in the extremely adversarial environment has been lacking. To close this research gap, we categorize existing attack techniques into three types based on the CIA triad: poisoning attacks (integrity \& availability infringement), evasion attacks (integrity infringement), and privacy attacks (confidentiality infringement). We have collected so far the most comprehensive (79) papers related to adversarial machine learning for CLM from the research fields of artificial intelligence, computer security, and software engineering. Our analysis covers each type of risk, examining threat model categorization, attack techniques, and countermeasures, while also introducing novel perspectives on eXplainable AI (XAI) and exploring the interconnections between different risks. Finally, we identify current challenges and future research opportunities. This study aims to provide a comprehensive roadmap for both researchers and practitioners and pave the way towards more reliable CLMs for practical applications.



## **8. Graph Agent Network: Empowering Nodes with Inference Capabilities for Adversarial Resilience**

cs.LG

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2306.06909v4) [paper-pdf](http://arxiv.org/pdf/2306.06909v4)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.



## **9. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Proactive Adversarial Defense**

cs.CR

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2407.15524v4) [paper-pdf](http://arxiv.org/pdf/2407.15524v4)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy due to its sensitivity to adversarial attacks. Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, existing anti-adversarial methods typically counteract adversarial perturbations post-attack, while we have devised a proactive strategy that preempts by safeguarding media upfront, effectively neutralizing potential adversarial effects before the third-party attacks occur. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first, to our knowledge, effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks.



## **10. Rapid Response: Mitigating LLM Jailbreaks with a Few Examples**

cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07494v1) [paper-pdf](http://arxiv.org/pdf/2411.07494v1)

**Authors**: Alwin Peng, Julian Michael, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: As large language models (LLMs) grow more powerful, ensuring their safety against misuse becomes crucial. While researchers have focused on developing robust defenses, no method has yet achieved complete invulnerability to attacks. We propose an alternative approach: instead of seeking perfect adversarial robustness, we develop rapid response techniques to look to block whole classes of jailbreaks after observing only a handful of attacks. To study this setting, we develop RapidResponseBench, a benchmark that measures a defense's robustness against various jailbreak strategies after adapting to a few observed examples. We evaluate five rapid response methods, all of which use jailbreak proliferation, where we automatically generate additional jailbreaks similar to the examples observed. Our strongest method, which fine-tunes an input classifier to block proliferated jailbreaks, reduces attack success rate by a factor greater than 240 on an in-distribution set of jailbreaks and a factor greater than 15 on an out-of-distribution set, having observed just one example of each jailbreaking strategy. Moreover, further studies suggest that the quality of proliferation model and number of proliferated examples play an key role in the effectiveness of this defense. Overall, our results highlight the potential of responding rapidly to novel jailbreaks to limit LLM misuse.



## **11. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

cs.CR

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.16914v3) [paper-pdf](http://arxiv.org/pdf/2402.16914v3)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.



## **12. The Inherent Adversarial Robustness of Analog In-Memory Computing**

cs.ET

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.07023v1) [paper-pdf](http://arxiv.org/pdf/2411.07023v1)

**Authors**: Corey Lammie, Julian Büchel, Athanasios Vasilopoulos, Manuel Le Gallo, Abu Sebastian

**Abstract**: A key challenge for Deep Neural Network (DNN) algorithms is their vulnerability to adversarial attacks. Inherently non-deterministic compute substrates, such as those based on Analog In-Memory Computing (AIMC), have been speculated to provide significant adversarial robustness when performing DNN inference. In this paper, we experimentally validate this conjecture for the first time on an AIMC chip based on Phase Change Memory (PCM) devices. We demonstrate higher adversarial robustness against different types of adversarial attacks when implementing an image classification network. Additional robustness is also observed when performing hardware-in-the-loop attacks, for which the attacker is assumed to have full access to the hardware. A careful study of the various noise sources indicate that a combination of stochastic noise sources (both recurrent and non-recurrent) are responsible for the adversarial robustness and that their type and magnitude disproportionately effects this property. Finally, it is demonstrated, via simulations, that when a much larger transformer network is used to implement a Natural Language Processing (NLP) task, additional robustness is still observed.



## **13. Computable Model-Independent Bounds for Adversarial Quantum Machine Learning**

cs.LG

21 pages, 9 figures

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06863v1) [paper-pdf](http://arxiv.org/pdf/2411.06863v1)

**Authors**: Bacui Li, Tansu Alpcan, Chandra Thapa, Udaya Parampalli

**Abstract**: By leveraging the principles of quantum mechanics, QML opens doors to novel approaches in machine learning and offers potential speedup. However, machine learning models are well-documented to be vulnerable to malicious manipulations, and this susceptibility extends to the models of QML. This situation necessitates a thorough understanding of QML's resilience against adversarial attacks, particularly in an era where quantum computing capabilities are expanding. In this regard, this paper examines model-independent bounds on adversarial performance for QML. To the best of our knowledge, we introduce the first computation of an approximate lower bound for adversarial error when evaluating model resilience against sophisticated quantum-based adversarial attacks. Experimental results are compared to the computed bound, demonstrating the potential of QML models to achieve high robustness. In the best case, the experimental error is only 10% above the estimated bound, offering evidence of the inherent robustness of quantum models. This work not only advances our theoretical understanding of quantum model resilience but also provides a precise reference bound for the future development of robust QML algorithms.



## **14. Boosting the Targeted Transferability of Adversarial Examples via Salient Region & Weighted Feature Drop**

cs.IR

9 pages

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06784v1) [paper-pdf](http://arxiv.org/pdf/2411.06784v1)

**Authors**: Shanjun Xu, Linghui Li, Kaiguo Yuan, Bingyu Li

**Abstract**: Deep neural networks can be vulnerable to adversarially crafted examples, presenting significant risks to practical applications. A prevalent approach for adversarial attacks relies on the transferability of adversarial examples, which are generated from a substitute model and leveraged to attack unknown black-box models. Despite various proposals aimed at improving transferability, the success of these attacks in targeted black-box scenarios is often hindered by the tendency for adversarial examples to overfit to the surrogate models. In this paper, we introduce a novel framework based on Salient region & Weighted Feature Drop (SWFD) designed to enhance the targeted transferability of adversarial examples. Drawing from the observation that examples with higher transferability exhibit smoother distributions in the deep-layer outputs, we propose the weighted feature drop mechanism to modulate activation values according to weights scaled by norm distribution, effectively addressing the overfitting issue when generating adversarial examples. Additionally, by leveraging salient region within the image to construct auxiliary images, our method enables the adversarial example's features to be transferred to the target category in a model-agnostic manner, thereby enhancing the transferability. Comprehensive experiments confirm that our approach outperforms state-of-the-art methods across diverse configurations. On average, the proposed SWFD raises the attack success rate for normally trained models and robust models by 16.31% and 7.06% respectively.



## **15. Beyond Text: Utilizing Vocal Cues to Improve Decision Making in LLMs for Robot Navigation Tasks**

cs.AI

30 pages, 7 figures

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.03494v3) [paper-pdf](http://arxiv.org/pdf/2402.03494v3)

**Authors**: Xingpeng Sun, Haoming Meng, Souradip Chakraborty, Amrit Singh Bedi, Aniket Bera

**Abstract**: While LLMs excel in processing text in these human conversations, they struggle with the nuances of verbal instructions in scenarios like social navigation, where ambiguity and uncertainty can erode trust in robotic and other AI systems. We can address this shortcoming by moving beyond text and additionally focusing on the paralinguistic features of these audio responses. These features are the aspects of spoken communication that do not involve the literal wording (lexical content) but convey meaning and nuance through how something is said. We present Beyond Text: an approach that improves LLM decision-making by integrating audio transcription along with a subsection of these features, which focus on the affect and more relevant in human-robot conversations.This approach not only achieves a 70.26% winning rate, outperforming existing LLMs by 22.16% to 48.30% (gemini-1.5-pro and gpt-3.5 respectively), but also enhances robustness against token manipulation adversarial attacks, highlighted by a 22.44% less decrease ratio than the text-only language model in winning rate. Beyond Text' marks an advancement in social robot navigation and broader Human-Robot interactions, seamlessly integrating text-based guidance with human-audio-informed language models.



## **16. Adversarial Detection with a Dynamically Stable System**

cs.AI

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06666v1) [paper-pdf](http://arxiv.org/pdf/2411.06666v1)

**Authors**: Xiaowei Long, Jie Lin, Xiangyuan Yang

**Abstract**: Adversarial detection is designed to identify and reject maliciously crafted adversarial examples(AEs) which are generated to disrupt the classification of target models.   Presently, various input transformation-based methods have been developed on adversarial example detection, which typically rely on empirical experience and lead to unreliability against new attacks.   To address this issue, we propose and conduct a Dynamically Stable System (DSS), which can effectively detect the adversarial examples from normal examples according to the stability of input examples.   Particularly, in our paper, the generation of adversarial examples is considered as the perturbation process of a Lyapunov dynamic system, and we propose an example stability mechanism, in which a novel control term is added in adversarial example generation to ensure that the normal examples can achieve dynamic stability while the adversarial examples cannot achieve the stability.   Then, based on the proposed example stability mechanism, a Dynamically Stable System (DSS) is proposed, which can utilize the disruption and restoration actions to determine the stability of input examples and detect the adversarial examples through changes in the stability of the input examples.   In comparison with existing methods in three benchmark datasets(MNIST, CIFAR10, and CIFAR100), our evaluation results show that our proposed DSS can achieve ROC-AUC values of 99.83%, 97.81% and 94.47%, surpassing the state-of-the-art(SOTA) values of 97.35%, 91.10% and 93.49% in the other 7 methods.



## **17. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2410.08827v2) [paper-pdf](http://arxiv.org/pdf/2410.08827v2)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.



## **18. HidePrint: Hiding the Radio Fingerprint via Random Noise**

cs.CR

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2411.06417v1) [paper-pdf](http://arxiv.org/pdf/2411.06417v1)

**Authors**: Gabriele Oligeri, Savio Sciancalepore

**Abstract**: Radio Frequency Fingerprinting (RFF) techniques allow a receiver to authenticate a transmitter by analyzing the physical layer of the radio spectrum. Although the vast majority of scientific contributions focus on improving the performance of RFF considering different parameters and scenarios, in this work, we consider RFF as an attack vector to identify and track a target device.   We propose, implement, and evaluate HidePrint, a solution to prevent tracking through RFF without affecting the quality of the communication link between the transmitter and the receiver. HidePrint hides the transmitter's fingerprint against an illegitimate eavesdropper by injecting controlled noise in the transmitted signal. We evaluate our solution against state-of-the-art image-based RFF techniques considering different adversarial models, different communication links (wired and wireless), and different configurations. Our results show that the injection of a Gaussian noise pattern with a standard deviation of (at least) 0.02 prevents device fingerprinting in all the considered scenarios, thus making the performance of the identification process indistinguishable from the random guess while affecting the Signal-to-Noise Ratio (SNR) of the received signal by only 0.1 dB. Moreover, we introduce selective radio fingerprint disclosure, a new technique that allows the transmitter to disclose the radio fingerprint to only a subset of intended receivers. This technique allows the transmitter to regain anonymity, thus preventing identification and tracking while allowing authorized receivers to authenticate the transmitter without affecting the quality of the transmitted signal.



## **19. Randomized Message-Interception Smoothing: Gray-box Certificates for Graph Neural Networks**

cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2301.02039v2) [paper-pdf](http://arxiv.org/pdf/2301.02039v2)

**Authors**: Yan Scholten, Jan Schuchardt, Simon Geisler, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Randomized smoothing is one of the most promising frameworks for certifying the adversarial robustness of machine learning models, including Graph Neural Networks (GNNs). Yet, existing randomized smoothing certificates for GNNs are overly pessimistic since they treat the model as a black box, ignoring the underlying architecture. To remedy this, we propose novel gray-box certificates that exploit the message-passing principle of GNNs: We randomly intercept messages and carefully analyze the probability that messages from adversarially controlled nodes reach their target nodes. Compared to existing certificates, we certify robustness to much stronger adversaries that control entire nodes in the graph and can arbitrarily manipulate node features. Our certificates provide stronger guarantees for attacks at larger distances, as messages from farther-away nodes are more likely to get intercepted. We demonstrate the effectiveness of our method on various models and datasets. Since our gray-box certificates consider the underlying graph structure, we can significantly improve certifiable robustness by applying graph sparsification.



## **20. Robust Detection of LLM-Generated Text: A Comparative Analysis**

cs.CL

8 pages

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06248v1) [paper-pdf](http://arxiv.org/pdf/2411.06248v1)

**Authors**: Yongye Su, Yuqing Wu

**Abstract**: The ability of large language models to generate complex texts allows them to be widely integrated into many aspects of life, and their output can quickly fill all network resources. As the impact of LLMs grows, it becomes increasingly important to develop powerful detectors for the generated text. This detector is essential to prevent the potential misuse of these technologies and to protect areas such as social media from the negative effects of false content generated by LLMS. The main goal of LLM-generated text detection is to determine whether text is generated by an LLM, which is a basic binary classification task. In our work, we mainly use three different classification methods based on open source datasets: traditional machine learning techniques such as logistic regression, k-means clustering, Gaussian Naive Bayes, support vector machines, and methods based on converters such as BERT, and finally algorithms that use LLMs to detect LLM-generated text. We focus on model generalization, potential adversarial attacks, and accuracy of model evaluation. Finally, the possible research direction in the future is proposed, and the current experimental results are summarized.



## **21. Target-driven Attack for Large Language Models**

cs.CL

12 pages, 7 figures. arXiv admin note: substantial text overlap with  arXiv:2404.07234

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.07268v1) [paper-pdf](http://arxiv.org/pdf/2411.07268v1)

**Authors**: Chong Zhang, Mingyu Jin, Dong Shu, Taowen Wang, Dongfang Liu, Xiaobo Jin

**Abstract**: Current large language models (LLM) provide a strong foundation for large-scale user-oriented natural language tasks. Many users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges like the language model not giving the correct answer. Although there is currently a large amount of research on black-box attacks, most of these black-box attacks use random and heuristic strategies. It is unclear how these strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we propose our target-driven black-box attack method to maximize the KL divergence between the conditional probabilities of the clean text and the attack text to redefine the attack's goal. We transform the distance maximization problem into two convex optimization problems based on the attack goal to solve the attack text and estimate the covariance. Furthermore, the projected gradient descent algorithm solves the vector corresponding to the attack text. Our target-driven black-box attack approach includes two attack strategies: token manipulation and misinformation attack. Experimental results on multiple Large Language Models and datasets demonstrate the effectiveness of our attack method.



## **22. BM-PAW: A Profitable Mining Attack in the PoW-based Blockchain System**

cs.CR

21 pages, 4 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06187v1) [paper-pdf](http://arxiv.org/pdf/2411.06187v1)

**Authors**: Junjie Hu, Xunzhi Chen, Huan Yan, Na Ruan

**Abstract**: Mining attacks enable an adversary to procure a disproportionately large portion of mining rewards by deviating from honest mining practices within the PoW-based blockchain system. In this paper, we demonstrate that the security vulnerabilities of PoW-based blockchain extend beyond what these mining attacks initially reveal. We introduce a novel mining strategy, named BM-PAW, which yields superior rewards for both the attacker and the targeted pool compared to the state-of-the-art mining attack: PAW. Our analysis reveals that BM-PAW attackers are incentivized to offer appropriate bribe money to other targets, as they comply with the attacker's directives upon receiving payment. We find the BM-PAW attacker can circumvent the "miner's dilemma" through equilibrium analysis in a two-pool BM-PAW game scenario, wherein the outcome is determined by the attacker's mining power. We finally propose practical countermeasures to mitigate these novel pool attacks.



## **23. AI-Compass: A Comprehensive and Effective Multi-module Testing Tool for AI Systems**

cs.AI

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06146v1) [paper-pdf](http://arxiv.org/pdf/2411.06146v1)

**Authors**: Zhiyu Zhu, Zhibo Jin, Hongsheng Hu, Minhui Xue, Ruoxi Sun, Seyit Camtepe, Praveen Gauravaram, Huaming Chen

**Abstract**: AI systems, in particular with deep learning techniques, have demonstrated superior performance for various real-world applications. Given the need for tailored optimization in specific scenarios, as well as the concerns related to the exploits of subsurface vulnerabilities, a more comprehensive and in-depth testing AI system becomes a pivotal topic. We have seen the emergence of testing tools in real-world applications that aim to expand testing capabilities. However, they often concentrate on ad-hoc tasks, rendering them unsuitable for simultaneously testing multiple aspects or components. Furthermore, trustworthiness issues arising from adversarial attacks and the challenge of interpreting deep learning models pose new challenges for developing more comprehensive and in-depth AI system testing tools. In this study, we design and implement a testing tool, \tool, to comprehensively and effectively evaluate AI systems. The tool extensively assesses multiple measurements towards adversarial robustness, model interpretability, and performs neuron analysis. The feasibility of the proposed testing tool is thoroughly validated across various modalities, including image classification, object detection, and text classification. Extensive experiments demonstrate that \tool is the state-of-the-art tool for a comprehensive assessment of the robustness and trustworthiness of AI systems. Our research sheds light on a general solution for AI systems testing landscape.



## **24. Robust Graph Neural Networks via Unbiased Aggregation**

cs.LG

NeurIPS 2024 poster. 28 pages, 14 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2311.14934v2) [paper-pdf](http://arxiv.org/pdf/2311.14934v2)

**Authors**: Zhichao Hou, Ruiqi Feng, Tyler Derr, Xiaorui Liu

**Abstract**: The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses. In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations. Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator. We then develop an efficient Quasi-Newton Iterative Reweighted Least Squares algorithm to solve the estimation problem, which is unfolded as robust unbiased aggregation layers in GNNs with theoretical guarantees. Our comprehensive experiments confirm the strong robustness of our proposed model under various scenarios, and the ablation study provides a deep understanding of its advantages. Our code is available at https://github.com/chris-hzc/RUNG.



## **25. Goal-guided Generative Prompt Injection Attack on Large Language Models**

cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2404.07234v4) [paper-pdf](http://arxiv.org/pdf/2404.07234v4)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.



## **26. Towards More Realistic Extraction Attacks: An Adversarial Perspective**

cs.CR

Presented at PrivateNLP@ACL2024

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2407.02596v2) [paper-pdf](http://arxiv.org/pdf/2407.02596v2)

**Authors**: Yash More, Prakhar Ganesh, Golnoosh Farnadi

**Abstract**: Language models are prone to memorizing parts of their training data which makes them vulnerable to extraction attacks. Existing research often examines isolated setups--such as evaluating extraction risks from a single model or with a fixed prompt design. However, a real-world adversary could access models across various sizes and checkpoints, as well as exploit prompt sensitivity, resulting in a considerably larger attack surface than previously studied. In this paper, we revisit extraction attacks from an adversarial perspective, focusing on how to leverage the brittleness of language models and the multi-faceted access to the underlying data. We find significant churn in extraction trends, i.e., even unintuitive changes to the prompt, or targeting smaller models and earlier checkpoints, can extract distinct information. By combining information from multiple attacks, our adversary is able to increase the extraction risks by up to $2 \times$. Furthermore, even with mitigation strategies like data deduplication, we find the same escalation of extraction risks against a real-world adversary. We conclude with a set of case studies, including detecting pre-training data, copyright violations, and extracting personally identifiable information, showing how our more realistic adversary can outperform existing adversaries in the literature.



## **27. A Survey of AI-Related Cyber Security Risks and Countermeasures in Mobility-as-a-Service**

cs.CR

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05681v1) [paper-pdf](http://arxiv.org/pdf/2411.05681v1)

**Authors**: Kai-Fung Chu, Haiyue Yuan, Jinsheng Yuan, Weisi Guo, Nazmiye Balta-Ozkan, Shujun Li

**Abstract**: Mobility-as-a-Service (MaaS) integrates different transport modalities and can support more personalisation of travellers' journey planning based on their individual preferences, behaviours and wishes. To fully achieve the potential of MaaS, a range of AI (including machine learning and data mining) algorithms are needed to learn personal requirements and needs, to optimise journey planning of each traveller and all travellers as a whole, to help transport service operators and relevant governmental bodies to operate and plan their services, and to detect and prevent cyber attacks from various threat actors including dishonest and malicious travellers and transport operators. The increasing use of different AI and data processing algorithms in both centralised and distributed settings opens the MaaS ecosystem up to diverse cyber and privacy attacks at both the AI algorithm level and the connectivity surfaces. In this paper, we present the first comprehensive review on the coupling between AI-driven MaaS design and the diverse cyber security challenges related to cyber attacks and countermeasures. In particular, we focus on how current and emerging AI-facilitated privacy risks (profiling, inference, and third-party threats) and adversarial AI attacks (evasion, extraction, and gamification) may impact the MaaS ecosystem. These risks often combine novel attacks (e.g., inverse learning) with traditional attack vectors (e.g., man-in-the-middle attacks), exacerbating the risks for the wider participation actors and the emergence of new business models.



## **28. DeepDRK: Deep Dependency Regularized Knockoff for Feature Selection**

cs.LG

33 pages, 15 figures, 9 tables

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2402.17176v2) [paper-pdf](http://arxiv.org/pdf/2402.17176v2)

**Authors**: Hongyu Shen, Yici Yan, Zhizhen Zhao

**Abstract**: Model-X knockoff has garnered significant attention among various feature selection methods due to its guarantees for controlling the false discovery rate (FDR). Since its introduction in parametric design, knockoff techniques have evolved to handle arbitrary data distributions using deep learning-based generative models. However, we have observed limitations in the current implementations of the deep Model-X knockoff framework. Notably, the "swap property" that knockoffs require often faces challenges at the sample level, resulting in diminished selection power. To address these issues, we develop "Deep Dependency Regularized Knockoff (DeepDRK)," a distribution-free deep learning method that effectively balances FDR and power. In DeepDRK, we introduce a novel formulation of the knockoff model as a learning problem under multi-source adversarial attacks. By employing an innovative perturbation technique, we achieve lower FDR and higher power. Our model outperforms existing benchmarks across synthetic, semi-synthetic, and real-world datasets, particularly when sample sizes are small and data distributions are non-Gaussian.



## **29. Towards a Re-evaluation of Data Forging Attacks in Practice**

cs.CR

18 pages

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05658v1) [paper-pdf](http://arxiv.org/pdf/2411.05658v1)

**Authors**: Mohamed Suliman, Anisa Halimi, Swanand Kadhe, Nathalie Baracaldo, Douglas Leith

**Abstract**: Data forging attacks provide counterfactual proof that a model was trained on a given dataset, when in fact, it was trained on another. These attacks work by forging (replacing) mini-batches with ones containing distinct training examples that produce nearly identical gradients. Data forging appears to break any potential avenues for data governance, as adversarial model owners may forge their training set from a dataset that is not compliant to one that is. Given these serious implications on data auditing and compliance, we critically analyse data forging from both a practical and theoretical point of view, finding that a key practical limitation of current attack methods makes them easily detectable by a verifier; namely that they cannot produce sufficiently identical gradients. Theoretically, we analyse the question of whether two distinct mini-batches can produce the same gradient. Generally, we find that while there may exist an infinite number of distinct mini-batches with real-valued training examples and labels that produce the same gradient, finding those that are within the allowed domain e.g. pixel values between 0-255 and one hot labels is a non trivial task. Our results call for the reevaluation of the strength of existing attacks, and for additional research into successful data forging, given the serious consequences it may have on machine learning and privacy.



## **30. BAN: Detecting Backdoors Activated by Adversarial Neuron Noise**

cs.LG

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2405.19928v2) [paper-pdf](http://arxiv.org/pdf/2405.19928v2)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Shujian Yu, Stjepan Picek

**Abstract**: Backdoor attacks on deep learning represent a recent threat that has gained significant attention in the research community. Backdoor defenses are mainly based on backdoor inversion, which has been shown to be generic, model-agnostic, and applicable to practical threat scenarios. State-of-the-art backdoor inversion recovers a mask in the feature space to locate prominent backdoor features, where benign and backdoor features can be disentangled. However, it suffers from high computational overhead, and we also find that it overly relies on prominent backdoor features that are highly distinguishable from benign features. To tackle these shortcomings, this paper improves backdoor feature inversion for backdoor detection by incorporating extra neuron activation information. In particular, we adversarially increase the loss of backdoored models with respect to weights to activate the backdoor effect, based on which we can easily differentiate backdoored and clean models. Experimental results demonstrate our defense, BAN, is 1.37$\times$ (on CIFAR-10) and 5.11$\times$ (on ImageNet200) more efficient with an average 9.99\% higher detect success rate than the state-of-the-art defense BTI-DBF. Our code and trained models are publicly available at~\url{https://github.com/xiaoyunxxy/ban}.



## **31. Post-Hoc Robustness Enhancement in Graph Neural Networks with Conditional Random Fields**

cs.LG

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05399v1) [paper-pdf](http://arxiv.org/pdf/2411.05399v1)

**Authors**: Yassine Abbahaddou, Sofiane Ennadir, Johannes F. Lutzeyer, Fragkiskos D. Malliaros, Michalis Vazirgiannis

**Abstract**: Graph Neural Networks (GNNs), which are nowadays the benchmark approach in graph representation learning, have been shown to be vulnerable to adversarial attacks, raising concerns about their real-world applicability. While existing defense techniques primarily concentrate on the training phase of GNNs, involving adjustments to message passing architectures or pre-processing methods, there is a noticeable gap in methods focusing on increasing robustness during inference. In this context, this study introduces RobustCRF, a post-hoc approach aiming to enhance the robustness of GNNs at the inference stage. Our proposed method, founded on statistical relational learning using a Conditional Random Field, is model-agnostic and does not require prior knowledge about the underlying model architecture. We validate the efficacy of this approach across various models, leveraging benchmark node classification datasets.



## **32. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

cs.LG

NeurIPS 2024 Spotlight; code available at  https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2401.17263v5) [paper-pdf](http://arxiv.org/pdf/2401.17263v5)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo



## **33. Region-Guided Attack on the Segment Anything Model (SAM)**

cs.CV

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.02974v2) [paper-pdf](http://arxiv.org/pdf/2411.02974v2)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao

**Abstract**: The Segment Anything Model (SAM) is a cornerstone of image segmentation, demonstrating exceptional performance across various applications, particularly in autonomous driving and medical imaging, where precise segmentation is crucial. However, SAM is vulnerable to adversarial attacks that can significantly impair its functionality through minor input perturbations. Traditional techniques, such as FGSM and PGD, are often ineffective in segmentation tasks due to their reliance on global perturbations that overlook spatial nuances. Recent methods like Attack-SAM-K and UAD have begun to address these challenges, but they frequently depend on external cues and do not fully leverage the structural interdependencies within segmentation processes. This limitation underscores the need for a novel adversarial strategy that exploits the unique characteristics of segmentation tasks. In response, we introduce the Region-Guided Attack (RGA), designed specifically for SAM. RGA utilizes a Region-Guided Map (RGM) to manipulate segmented regions, enabling targeted perturbations that fragment large segments and expand smaller ones, resulting in erroneous outputs from SAM. Our experiments demonstrate that RGA achieves high success rates in both white-box and black-box scenarios, emphasizing the need for robust defenses against such sophisticated attacks. RGA not only reveals SAM's vulnerabilities but also lays the groundwork for developing more resilient defenses against adversarial threats in image segmentation.



## **34. Accelerating Greedy Coordinate Gradient and General Prompt Optimization via Probe Sampling**

cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2403.01251v3) [paper-pdf](http://arxiv.org/pdf/2403.01251v3)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a critical issue given their rapid progresses. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing adversarial prompts to break the aligned LLMs, but optimization of GCG is time-consuming. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b-chat and leads to equal or improved attack success rate (ASR) on the AdvBench. Furthermore, probe sampling is also able to accelerate other prompt optimization techniques and adversarial methods, leading to acceleration of $1.8\times$ for AutoPrompt, $2.4\times$ for APE and $2.4\times$ for AutoDAN.



## **35. Reasoning Robustness of LLMs to Adversarial Typographical Errors**

cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05345v1) [paper-pdf](http://arxiv.org/pdf/2411.05345v1)

**Authors**: Esther Gan, Yiran Zhao, Liying Cheng, Yancan Mao, Anirudh Goyal, Kenji Kawaguchi, Min-Yen Kan, Michael Shieh

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning using Chain-of-Thought (CoT) prompting. However, CoT can be biased by users' instruction. In this work, we study the reasoning robustness of LLMs to typographical errors, which can naturally occur in users' queries. We design an Adversarial Typo Attack ($\texttt{ATA}$) algorithm that iteratively samples typos for words that are important to the query and selects the edit that is most likely to succeed in attacking. It shows that LLMs are sensitive to minimal adversarial typographical changes. Notably, with 1 character edit, Mistral-7B-Instruct's accuracy drops from 43.7% to 38.6% on GSM8K, while with 8 character edits the performance further drops to 19.2%. To extend our evaluation to larger and closed-source LLMs, we develop the $\texttt{R$^2$ATA}$ benchmark, which assesses models' $\underline{R}$easoning $\underline{R}$obustness to $\underline{\texttt{ATA}}$. It includes adversarial typographical questions derived from three widely used reasoning datasets-GSM8K, BBH, and MMLU-by applying $\texttt{ATA}$ to open-source LLMs. $\texttt{R$^2$ATA}$ demonstrates remarkable transferability and causes notable performance drops across multiple super large and closed-source LLMs.



## **36. Fight Fire with Fire: Combating Adversarial Patch Attacks using Pattern-randomized Defensive Patches**

cs.CV

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2311.06122v2) [paper-pdf](http://arxiv.org/pdf/2311.06122v2)

**Authors**: Jianan Feng, Jiachun Li, Changqing Miao, Jianjun Huang, Wei You, Wenchang Shi, Bin Liang

**Abstract**: Object detection has found extensive applications in various tasks, but it is also susceptible to adversarial patch attacks. The ideal defense should be effective, efficient, easy to deploy, and capable of withstanding adaptive attacks. In this paper, we adopt a counterattack strategy to propose a novel and general methodology for defending adversarial attacks. Two types of defensive patches, canary and woodpecker, are specially-crafted and injected into the model input to proactively probe or counteract potential adversarial patches. In this manner, adversarial patch attacks can be effectively detected by simply analyzing the model output, without the need to alter the target model. Moreover, we employ randomized canary and woodpecker injection patterns to defend against defense-aware attacks. The effectiveness and practicality of the proposed method are demonstrated through comprehensive experiments. The results illustrate that canary and woodpecker achieve high performance, even when confronted with unknown attack methods, while incurring limited time overhead. Furthermore, our method also exhibits sufficient robustness against defense-aware attacks, as evidenced by adaptive attack experiments.



## **37. Towards Secured Smart Grid 2.0: Exploring Security Threats, Protection Models, and Challenges**

cs.NI

30 pages, 21 figures, 5 tables, accepted to appear in IEEE COMST

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.04365v2) [paper-pdf](http://arxiv.org/pdf/2411.04365v2)

**Authors**: Lan-Huong Nguyen, Van-Linh Nguyen, Ren-Hung Hwang, Jian-Jhih Kuo, Yu-Wen Chen, Chien-Chung Huang, Ping-I Pan

**Abstract**: Many nations are promoting the green transition in the energy sector to attain neutral carbon emissions by 2050. Smart Grid 2.0 (SG2) is expected to explore data-driven analytics and enhance communication technologies to improve the efficiency and sustainability of distributed renewable energy systems. These features are beyond smart metering and electric surplus distribution in conventional smart grids. Given the high dependence on communication networks to connect distributed microgrids in SG2, potential cascading failures of connectivity can cause disruption to data synchronization to the remote control systems. This paper reviews security threats and defense tactics for three stakeholders: power grid operators, communication network providers, and consumers. Through the survey, we found that SG2's stakeholders are particularly vulnerable to substation attacks/vandalism, malware/ransomware threats, blockchain vulnerabilities and supply chain breakdowns. Furthermore, incorporating artificial intelligence (AI) into autonomous energy management in distributed energy resources of SG2 creates new challenges. Accordingly, adversarial samples and false data injection on electricity reading and measurement sensors at power plants can fool AI-powered control functions and cause messy error-checking operations in energy storage, wrong energy estimation in electric vehicle charging, and even fraudulent transactions in peer-to-peer energy trading models. Scalable blockchain-based models, physical unclonable function, interoperable security protocols, and trustworthy AI models designed for managing distributed microgrids in SG2 are typical promising protection models for future research.



## **38. Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling**

cs.CV

Accepted by CVPR 2023

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2307.01778v2) [paper-pdf](http://arxiv.org/pdf/2307.01778v2)

**Authors**: Zhanhao Hu, Wenda Chu, Xiaopei Zhu, Hui Zhang, Bo Zhang, Xiaolin Hu

**Abstract**: Recent works have proposed to craft adversarial clothes for evading person detectors, while they are either only effective at limited viewing angles or very conspicuous to humans. We aim to craft adversarial texture for clothes based on 3D modeling, an idea that has been used to craft rigid adversarial objects such as a 3D-printed turtle. Unlike rigid objects, humans and clothes are non-rigid, leading to difficulties in physical realization. In order to craft natural-looking adversarial clothes that can evade person detectors at multiple viewing angles, we propose adversarial camouflage textures (AdvCaT) that resemble one kind of the typical textures of daily clothes, camouflage textures. We leverage the Voronoi diagram and Gumbel-softmax trick to parameterize the camouflage textures and optimize the parameters via 3D modeling. Moreover, we propose an efficient augmentation pipeline on 3D meshes combining topologically plausible projection (TopoProj) and Thin Plate Spline (TPS) to narrow the gap between digital and real-world objects. We printed the developed 3D texture pieces on fabric materials and tailored them into T-shirts and trousers. Experiments show high attack success rates of these clothes against multiple detectors.



## **39. A Barrier Certificate-based Simplex Architecture for Systems with Approximate and Hybrid Dynamics**

eess.SY

This version includes the following new contributions. (1) We extend  Bb-Simplex to hybrid systems and prove the correctness of this extension. (2)  We extend Bb-Simplex to support the use of approximate dynamics. (3) We  combine these two extensions of Bb-Simplex. (4) We present new experiments  evaluating Bb-Simplex and its extensions using a complex model of a real  microgrid

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2202.09710v3) [paper-pdf](http://arxiv.org/pdf/2202.09710v3)

**Authors**: Amol Damare, Shouvik Roy, Roshan Sharma, Keith DSouza, Scott A. Smolka, Scott D. Stoller

**Abstract**: We present Barrier-based Simplex (Bb-Simplex), a new, provably correct design for runtime assurance of continuous dynamical systems. Bb-Simplex is centered around the Simplex control architecture, which consists of a high-performance advanced controller that is not guaranteed to maintain safety of the plant, a verified-safe baseline controller, and a decision module that switches control of the plant between the two controllers to ensure safety without sacrificing performance. In Bb-Simplex, Barrier certificates are used to prove that the baseline controller ensures safety. Furthermore, Bb-Simplex features a new automated method for deriving, from the barrier certificate, the conditions for switching between the controllers. Our method is based on the Taylor expansion of the barrier certificate and yields computationally inexpensive switching conditions.   We also propose extensions to Bb-Simplex to enable its use in hybrid systems, which have multiple modes each with its own dynamics, and to support its use when only approximate dynamics (not exact dynamics) are available, for both continuous-time and hybrid dynamical systems.   We consider significant applications of Bb-Simplex to microgrids featuring advanced controllers in the form of neural networks trained using reinforcement learning. These microgrids are modeled in RTDS, an industry-standard high-fidelity, real-time power systems simulator. Our results demonstrate that Bb-Simplex can automatically derive switching conditions for complex continuous-time and hybrid systems, the switching conditions are not overly conservative, and Bb-Simplex ensures safety even in the presence of adversarial attacks on the neural controller when only approximate dynamics (with an error bound) are available.



## **40. Adversarial Robustness of In-Context Learning in Transformers for Linear Regression**

cs.LG

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.05189v1) [paper-pdf](http://arxiv.org/pdf/2411.05189v1)

**Authors**: Usman Anwar, Johannes Von Oswald, Louis Kirsch, David Krueger, Spencer Frei

**Abstract**: Transformers have demonstrated remarkable in-context learning capabilities across various domains, including statistical learning tasks. While previous work has shown that transformers can implement common learning algorithms, the adversarial robustness of these learned algorithms remains unexplored. This work investigates the vulnerability of in-context learning in transformers to \textit{hijacking attacks} focusing on the setting of linear regression tasks. Hijacking attacks are prompt-manipulation attacks in which the adversary's goal is to manipulate the prompt to force the transformer to generate a specific output. We first prove that single-layer linear transformers, known to implement gradient descent in-context, are non-robust and can be manipulated to output arbitrary predictions by perturbing a single example in the in-context training set. While our experiments show these attacks succeed on linear transformers, we find they do not transfer to more complex transformers with GPT-2 architectures. Nonetheless, we show that these transformers can be hijacked using gradient-based adversarial attacks. We then demonstrate that adversarial training enhances transformers' robustness against hijacking attacks, even when just applied during finetuning. Additionally, we find that in some settings, adversarial training against a weaker attack model can lead to robustness to a stronger attack model. Lastly, we investigate the transferability of hijacking attacks across transformers of varying scales and initialization seeds, as well as between transformers and ordinary least squares (OLS). We find that while attacks transfer effectively between small-scale transformers, they show poor transferability in other scenarios (small-to-large scale, large-to-large scale, and between transformers and OLS).



## **41. Seeing is Deceiving: Exploitation of Visual Pathways in Multi-Modal Language Models**

cs.CR

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.05056v1) [paper-pdf](http://arxiv.org/pdf/2411.05056v1)

**Authors**: Pete Janowczyk, Linda Laurier, Ave Giulietta, Arlo Octavia, Meade Cleti

**Abstract**: Multi-Modal Language Models (MLLMs) have transformed artificial intelligence by combining visual and text data, making applications like image captioning, visual question answering, and multi-modal content creation possible. This ability to understand and work with complex information has made MLLMs useful in areas such as healthcare, autonomous systems, and digital content. However, integrating multiple types of data also creates security risks. Attackers can manipulate either the visual or text inputs, or both, to make the model produce unintended or even harmful responses. This paper reviews how visual inputs in MLLMs can be exploited by various attack strategies. We break down these attacks into categories: simple visual tweaks and cross-modal manipulations, as well as advanced strategies like VLATTACK, HADES, and Collaborative Multimodal Adversarial Attack (Co-Attack). These attacks can mislead even the most robust models while looking nearly identical to the original visuals, making them hard to detect. We also discuss the broader security risks, including threats to privacy and safety in important applications. To counter these risks, we review current defense methods like the SmoothVLM framework, pixel-wise randomization, and MirrorCheck, looking at their strengths and limitations. We also discuss new methods to make MLLMs more secure, including adaptive defenses, better evaluation tools, and security approaches that protect both visual and text data. By bringing together recent developments and identifying key areas for improvement, this review aims to support the creation of more secure and reliable multi-modal AI systems for real-world use.



## **42. FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (Automated Multi-shot Jailbreaks)**

cs.CL

4 pages, 2 tables

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2408.16163v2) [paper-pdf](http://arxiv.org/pdf/2408.16163v2)

**Authors**: Aman Priyanshu, Supriti Vijay

**Abstract**: This paper introduces FRACTURED-SORRY-Bench, a framework for evaluating the safety of Large Language Models (LLMs) against multi-turn conversational attacks. Building upon the SORRY-Bench dataset, we propose a simple yet effective method for generating adversarial prompts by breaking down harmful queries into seemingly innocuous sub-questions. Our approach achieves a maximum increase of +46.22\% in Attack Success Rates (ASRs) across GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-Turbo models compared to baseline methods. We demonstrate that this technique poses a challenge to current LLM safety measures and highlights the need for more robust defenses against subtle, multi-turn attacks.



## **43. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

cs.CR

Accepted by NeurIPS 2024. Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2403.00867v3) [paper-pdf](http://arxiv.org/pdf/2403.00867v3)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.



## **44. Attention Masks Help Adversarial Attacks to Bypass Safety Detectors**

cs.CR

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04772v1) [paper-pdf](http://arxiv.org/pdf/2411.04772v1)

**Authors**: Yunfan Shi

**Abstract**: Despite recent research advancements in adversarial attack methods, current approaches against XAI monitors are still discoverable and slower. In this paper, we present an adaptive framework for attention mask generation to enable stealthy, explainable and efficient PGD image classification adversarial attack under XAI monitors. Specifically, we utilize mutation XAI mixture and multitask self-supervised X-UNet for attention mask generation to guide PGD attack. Experiments on MNIST (MLP), CIFAR-10 (AlexNet) have shown that our system can outperform benchmark PGD, Sparsefool and SOTA SINIFGSM in balancing among stealth, efficiency and explainability which is crucial for effectively fooling SOTA defense protected classifiers.



## **45. MISGUIDE: Security-Aware Attack Analytics for Smart Grid Load Frequency Control**

cs.CE

12 page journal

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04731v1) [paper-pdf](http://arxiv.org/pdf/2411.04731v1)

**Authors**: Nur Imtiazul Haque, Prabin Mali, Mohammad Zakaria Haider, Mohammad Ashiqur Rahman, Sumit Paudyal

**Abstract**: Incorporating advanced information and communication technologies into smart grids (SGs) offers substantial operational benefits while increasing vulnerability to cyber threats like false data injection (FDI) attacks. Current SG attack analysis tools predominantly employ formal methods or adversarial machine learning (ML) techniques with rule-based bad data detectors to analyze the attack space. However, these attack analytics either generate simplistic attack vectors detectable by the ML-based anomaly detection models (ADMs) or fail to identify critical attack vectors from complex controller dynamics in a feasible time. This paper introduces MISGUIDE, a novel defense-aware attack analytics designed to extract verifiable multi-time slot-based FDI attack vectors from complex SG load frequency control dynamics and ADMs, utilizing the Gurobi optimizer. MISGUIDE can identify optimal (maliciously triggering under/over frequency relays in minimal time) and stealthy attack vectors. Using real-world load data, we validate the MISGUIDE-identified attack vectors through real-time hardware-in-the-loop (OPALRT) simulations of the IEEE 39-bus system.



## **46. Neural Fingerprints for Adversarial Attack Detection**

cs.CV

14 pages

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04533v1) [paper-pdf](http://arxiv.org/pdf/2411.04533v1)

**Authors**: Haim Fisher, Moni Shahar, Yehezkel S. Resheff

**Abstract**: Deep learning models for image classification have become standard tools in recent years. A well known vulnerability of these models is their susceptibility to adversarial examples. These are generated by slightly altering an image of a certain class in a way that is imperceptible to humans but causes the model to classify it wrongly as another class. Many algorithms have been proposed to address this problem, falling generally into one of two categories: (i) building robust classifiers (ii) directly detecting attacked images. Despite the good performance of these detectors, we argue that in a white-box setting, where the attacker knows the configuration and weights of the network and the detector, they can overcome the detector by running many examples on a local copy, and sending only those that were not detected to the actual model. This problem is common in security applications where even a very good model is not sufficient to ensure safety. In this paper we propose to overcome this inherent limitation of any static defence with randomization. To do so, one must generate a very large family of detectors with consistent performance, and select one or more of them randomly for each input. For the individual detectors, we suggest the method of neural fingerprints. In the training phase, for each class we repeatedly sample a tiny random subset of neurons from certain layers of the network, and if their average is sufficiently different between clean and attacked images of the focal class they are considered a fingerprint and added to the detector bank. During test time, we sample fingerprints from the bank associated with the label predicted by the model, and detect attacks using a likelihood ratio test. We evaluate our detectors on ImageNet with different attack methods and model architectures, and show near-perfect detection with low rates of false detection.



## **47. Undermining Image and Text Classification Algorithms Using Adversarial Attacks**

cs.CR

Accepted for presentation at Electronic Imaging Conference 2025

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.03348v2) [paper-pdf](http://arxiv.org/pdf/2411.03348v2)

**Authors**: Langalibalele Lunga, Suhas Sreehari

**Abstract**: Machine learning models are prone to adversarial attacks, where inputs can be manipulated in order to cause misclassifications. While previous research has focused on techniques like Generative Adversarial Networks (GANs), there's limited exploration of GANs and Synthetic Minority Oversampling Technique (SMOTE) in text and image classification models to perform adversarial attacks. Our study addresses this gap by training various machine learning models and using GANs and SMOTE to generate additional data points aimed at attacking text classification models. Furthermore, we extend our investigation to face recognition models, training a Convolutional Neural Network(CNN) and subjecting it to adversarial attacks with fast gradient sign perturbations on key features identified by GradCAM, a technique used to highlight key image characteristics CNNs use in classification. Our experiments reveal a significant vulnerability in classification models. Specifically, we observe a 20 % decrease in accuracy for the top-performing text classification models post-attack, along with a 30 % decrease in facial recognition accuracy. This highlights the susceptibility of these models to manipulation of input data. Adversarial attacks not only compromise the security but also undermine the reliability of machine learning systems. By showcasing the impact of adversarial attacks on both text classification and face recognition models, our study underscores the urgent need for develop robust defenses against such vulnerabilities.



## **48. Game-Theoretic Defenses for Robust Conformal Prediction Against Adversarial Attacks in Medical Imaging**

cs.LG

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04376v1) [paper-pdf](http://arxiv.org/pdf/2411.04376v1)

**Authors**: Rui Luo, Jie Bao, Zhixin Zhou, Chuangyin Dang

**Abstract**: Adversarial attacks pose significant threats to the reliability and safety of deep learning models, especially in critical domains such as medical imaging. This paper introduces a novel framework that integrates conformal prediction with game-theoretic defensive strategies to enhance model robustness against both known and unknown adversarial perturbations. We address three primary research questions: constructing valid and efficient conformal prediction sets under known attacks (RQ1), ensuring coverage under unknown attacks through conservative thresholding (RQ2), and determining optimal defensive strategies within a zero-sum game framework (RQ3). Our methodology involves training specialized defensive models against specific attack types and employing maximum and minimum classifiers to aggregate defenses effectively. Extensive experiments conducted on the MedMNIST datasets, including PathMNIST, OrganAMNIST, and TissueMNIST, demonstrate that our approach maintains high coverage guarantees while minimizing prediction set sizes. The game-theoretic analysis reveals that the optimal defensive strategy often converges to a singular robust model, outperforming uniform and simple strategies across all evaluated datasets. This work advances the state-of-the-art in uncertainty quantification and adversarial robustness, providing a reliable mechanism for deploying deep learning models in adversarial environments.



## **49. $B^4$: A Black-Box Scrubbing Attack on LLM Watermarks**

cs.CL

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.01222v3) [paper-pdf](http://arxiv.org/pdf/2411.01222v3)

**Authors**: Baizhou Huang, Xiao Pu, Xiaojun Wan

**Abstract**: Watermarking has emerged as a prominent technique for LLM-generated content detection by embedding imperceptible patterns. Despite supreme performance, its robustness against adversarial attacks remains underexplored. Previous work typically considers a grey-box attack setting, where the specific type of watermark is already known. Some even necessitates knowledge about hyperparameters of the watermarking method. Such prerequisites are unattainable in real-world scenarios. Targeting at a more realistic black-box threat model with fewer assumptions, we here propose $B^4$, a black-box scrubbing attack on watermarks. Specifically, we formulate the watermark scrubbing attack as a constrained optimization problem by capturing its objectives with two distributions, a Watermark Distribution and a Fidelity Distribution. This optimization problem can be approximately solved using two proxy distributions. Experimental results across 12 different settings demonstrate the superior performance of $B^4$ compared with other baselines.



## **50. Transferable Learned Image Compression-Resistant Adversarial Perturbations**

cs.CV

Accepted by BMVC 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2401.03115v2) [paper-pdf](http://arxiv.org/pdf/2401.03115v2)

**Authors**: Yang Sui, Zhuohang Li, Ding Ding, Xiang Pan, Xiaozhong Xu, Shan Liu, Zhenzhong Chen

**Abstract**: Adversarial attacks can readily disrupt the image classification system, revealing the vulnerability of DNN-based recognition tasks. While existing adversarial perturbations are primarily applied to uncompressed images or compressed images by the traditional image compression method, i.e., JPEG, limited studies have investigated the robustness of models for image classification in the context of DNN-based image compression. With the rapid evolution of advanced image compression, DNN-based learned image compression has emerged as the promising approach for transmitting images in many security-critical applications, such as cloud-based face recognition and autonomous driving, due to its superior performance over traditional compression. Therefore, there is a pressing need to fully investigate the robustness of a classification system post-processed by learned image compression. To bridge this research gap, we explore the adversarial attack on a new pipeline that targets image classification models that utilize learned image compressors as pre-processing modules. Furthermore, to enhance the transferability of perturbations across various quality levels and architectures of learned image compression models, we introduce a saliency score-based sampling method to enable the fast generation of transferable perturbation. Extensive experiments with popular attack methods demonstrate the enhanced transferability of our proposed method when attacking images that have been post-processed with different learned image compression models.



