# Latest Adversarial Attack Papers
**update at 2023-12-22 18:51:02**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Open-Set: ID Card Presentation Attack Detection using Neural Transfer Style**

cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13993v1) [paper-pdf](http://arxiv.org/pdf/2312.13993v1)

**Authors**: Reuben Markham, Juan M. Espin, Mario Nieto-Hidalgo, Juan E. Tapia

**Abstract**: The accurate detection of ID card Presentation Attacks (PA) is becoming increasingly important due to the rising number of online/remote services that require the presentation of digital photographs of ID cards for digital onboarding or authentication. Furthermore, cybercriminals are continuously searching for innovative ways to fool authentication systems to gain unauthorized access to these services. Although advances in neural network design and training have pushed image classification to the state of the art, one of the main challenges faced by the development of fraud detection systems is the curation of representative datasets for training and evaluation. The handcrafted creation of representative presentation attack samples often requires expertise and is very time-consuming, thus an automatic process of obtaining high-quality data is highly desirable. This work explores ID card Presentation Attack Instruments (PAI) in order to improve the generation of samples with four Generative Adversarial Networks (GANs) based image translation models and analyses the effectiveness of the generated data for training fraud detection systems. Using open-source data, we show that synthetic attack presentations are an adequate complement for additional real attack presentations, where we obtain an EER performance increase of 0.63% points for print attacks and a loss of 0.29% for screen capture attacks.



## **2. Quantum Neural Networks under Depolarization Noise: Exploring White-Box Attacks and Defenses**

quant-ph

Poster at Quantum Techniques in Machine Learning (QTML) 2023

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2311.17458v2) [paper-pdf](http://arxiv.org/pdf/2311.17458v2)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: Leveraging the unique properties of quantum mechanics, Quantum Machine Learning (QML) promises computational breakthroughs and enriched perspectives where traditional systems reach their boundaries. However, similarly to classical machine learning, QML is not immune to adversarial attacks. Quantum adversarial machine learning has become instrumental in highlighting the weak points of QML models when faced with adversarial crafted feature vectors. Diving deep into this domain, our exploration shines light on the interplay between depolarization noise and adversarial robustness. While previous results enhanced robustness from adversarial threats through depolarization noise, our findings paint a different picture. Interestingly, adding depolarization noise discontinued the effect of providing further robustness for a multi-class classification scenario. Consolidating our findings, we conducted experiments with a multi-class classifier adversarially trained on gate-based quantum simulators, further elucidating this unexpected behavior.



## **3. Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples**

cs.LG

Accepted by AAAI-2024

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13628v1) [paper-pdf](http://arxiv.org/pdf/2312.13628v1)

**Authors**: Ruichu Cai, Yuxuan Zhu, Jie Qiao, Zefeng Liang, Furui Liu, Zhifeng Hao

**Abstract**: Deep neural networks (DNNs) have been demonstrated to be vulnerable to well-crafted \emph{adversarial examples}, which are generated through either well-conceived $\mathcal{L}_p$-norm restricted or unrestricted attacks. Nevertheless, the majority of those approaches assume that adversaries can modify any features as they wish, and neglect the causal generating process of the data, which is unreasonable and unpractical. For instance, a modification in income would inevitably impact features like the debt-to-income ratio within a banking system. By considering the underappreciated causal generating process, first, we pinpoint the source of the vulnerability of DNNs via the lens of causality, then give theoretical results to answer \emph{where to attack}. Second, considering the consequences of the attack interventions on the current state of the examples to generate more realistic adversarial examples, we propose CADE, a framework that can generate \textbf{C}ounterfactual \textbf{AD}versarial \textbf{E}xamples to answer \emph{how to attack}. The empirical results demonstrate CADE's effectiveness, as evidenced by its competitive performance across diverse attack scenarios, including white-box, transfer-based, and random intervention attacks.



## **4. ARBiBench: Benchmarking Adversarial Robustness of Binarized Neural Networks**

cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13575v1) [paper-pdf](http://arxiv.org/pdf/2312.13575v1)

**Authors**: Peng Zhao, Jiehua Zhang, Bowen Peng, Longguang Wang, YingMei Wei, Yu Liu, Li Liu

**Abstract**: Network binarization exhibits great potential for deployment on resource-constrained devices due to its low computational cost. Despite the critical importance, the security of binarized neural networks (BNNs) is rarely investigated. In this paper, we present ARBiBench, a comprehensive benchmark to evaluate the robustness of BNNs against adversarial perturbations on CIFAR-10 and ImageNet. We first evaluate the robustness of seven influential BNNs on various white-box and black-box attacks. The results reveal that 1) The adversarial robustness of BNNs exhibits a completely opposite performance on the two datasets under white-box attacks. 2) BNNs consistently exhibit better adversarial robustness under black-box attacks. 3) Different BNNs exhibit certain similarities in their robustness performance. Then, we conduct experiments to analyze the adversarial robustness of BNNs based on these insights. Our research contributes to inspiring future research on enhancing the robustness of BNNs and advancing their application in real-world scenarios.



## **5. Adversarial Purification with the Manifold Hypothesis**

cs.LG

Extended version of paper accepted at AAAI 2024 with supplementary  materials

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2210.14404v5) [paper-pdf](http://arxiv.org/pdf/2210.14404v5)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework for adversarial robustness using the manifold hypothesis. This framework provides sufficient conditions for defending against adversarial examples. We develop an adversarial purification method with this framework. Our method combines manifold learning with variational inference to provide adversarial robustness without the need for expensive adversarial training. Experimentally, our approach can provide adversarial robustness even if attackers are aware of the existence of the defense. In addition, our method can also serve as a test-time defense mechanism for variational autoencoders.



## **6. Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**

cs.AI

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13435v1) [paper-pdf](http://arxiv.org/pdf/2312.13435v1)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.



## **7. Universal and Transferable Adversarial Attacks on Aligned Language Models**

cs.CL

Website: http://llm-attacks.org/

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2307.15043v2) [paper-pdf](http://arxiv.org/pdf/2307.15043v2)

**Authors**: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.



## **8. On the complexity of sabotage games for network security**

cs.CC

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13132v1) [paper-pdf](http://arxiv.org/pdf/2312.13132v1)

**Authors**: Dhananjay Raju, Georgios Bakirtzis, Ufuk Topcu

**Abstract**: Securing dynamic networks against adversarial actions is challenging because of the need to anticipate and counter strategic disruptions by adversarial entities within complex network structures. Traditional game-theoretic models, while insightful, often fail to model the unpredictability and constraints of real-world threat assessment scenarios. We refine sabotage games to reflect the realistic limitations of the saboteur and the network operator. By transforming sabotage games into reachability problems, our approach allows applying existing computational solutions to model realistic restrictions on attackers and defenders within the game. Modifying sabotage games into dynamic network security problems successfully captures the nuanced interplay of strategy and uncertainty in dynamic network security. Theoretically, we extend sabotage games to model network security contexts and thoroughly explore if the additional restrictions raise their computational complexity, often the bottleneck of game theory in practical contexts. Practically, this research sets the stage for actionable insights for developing robust defense mechanisms by understanding what risks to mitigate in dynamically changing networks under threat.



## **9. Prometheus: Infrastructure Security Posture Analysis with AI-generated Attack Graphs**

cs.CR

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13119v1) [paper-pdf](http://arxiv.org/pdf/2312.13119v1)

**Authors**: Xin Jin, Charalampos Katsis, Fan Sang, Jiahao Sun, Elisa Bertino, Ramana Rao Kompella, Ashish Kundu

**Abstract**: The rampant occurrence of cybersecurity breaches imposes substantial limitations on the progress of network infrastructures, leading to compromised data, financial losses, potential harm to individuals, and disruptions in essential services. The current security landscape demands the urgent development of a holistic security assessment solution that encompasses vulnerability analysis and investigates the potential exploitation of these vulnerabilities as attack paths. In this paper, we propose Prometheus, an advanced system designed to provide a detailed analysis of the security posture of computing infrastructures. Using user-provided information, such as device details and software versions, Prometheus performs a comprehensive security assessment. This assessment includes identifying associated vulnerabilities and constructing potential attack graphs that adversaries can exploit. Furthermore, Prometheus evaluates the exploitability of these attack paths and quantifies the overall security posture through a scoring mechanism. The system takes a holistic approach by analyzing security layers encompassing hardware, system, network, and cryptography. Furthermore, Prometheus delves into the interconnections between these layers, exploring how vulnerabilities in one layer can be leveraged to exploit vulnerabilities in others. In this paper, we present the end-to-end pipeline implemented in Prometheus, showcasing the systematic approach adopted for conducting this thorough security analysis.



## **10. LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**

cs.LG

AAAI 2024

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13118v1) [paper-pdf](http://arxiv.org/pdf/2312.13118v1)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.



## **11. PGN: A perturbation generation network against deep reinforcement learning**

cs.LG

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.12904v1) [paper-pdf](http://arxiv.org/pdf/2312.12904v1)

**Authors**: Xiangjuan Li, Feifan Li, Yang Li, Quan Pan

**Abstract**: Deep reinforcement learning has advanced greatly and applied in many areas. In this paper, we explore the vulnerability of deep reinforcement learning by proposing a novel generative model for creating effective adversarial examples to attack the agent. Our proposed model can achieve both targeted attacks and untargeted attacks. Considering the specificity of deep reinforcement learning, we propose the action consistency ratio as a measure of stealthiness, and a new measurement index of effectiveness and stealthiness. Experiment results show that our method can ensure the effectiveness and stealthiness of attack compared with other algorithms. Moreover, our methods are considerably faster and thus can achieve rapid and efficient verification of the vulnerability of deep reinforcement learning.



## **12. SAAM: Stealthy Adversarial Attack on Monocular Depth Estimation**

cs.CV

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2308.03108v2) [paper-pdf](http://arxiv.org/pdf/2308.03108v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique

**Abstract**: In this paper, we investigate the vulnerability of MDE to adversarial patches. We propose a novel \underline{S}tealthy \underline{A}dversarial \underline{A}ttacks on \underline{M}DE (SAAM) that compromises MDE by either corrupting the estimated distance or causing an object to seamlessly blend into its surroundings. Our experiments, demonstrate that the designed stealthy patch successfully causes a DNN-based MDE to misestimate the depth of objects. In fact, our proposed adversarial patch achieves a significant 60\% depth error with 99\% ratio of the affected region. Importantly, despite its adversarial nature, the patch maintains a naturalistic appearance, making it inconspicuous to human observers. We believe that this work sheds light on the threat of adversarial attacks in the context of MDE on edge devices. We hope it raises awareness within the community about the potential real-life harm of such attacks and encourages further research into developing more robust and adaptive defense mechanisms.



## **13. Mutual-modality Adversarial Attack with Semantic Perturbation**

cs.CV

Accepted by AAAI2024

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.12768v1) [paper-pdf](http://arxiv.org/pdf/2312.12768v1)

**Authors**: Jingwen Ye, Ruonan Yu, Songhua Liu, Xinchao Wang

**Abstract**: Adversarial attacks constitute a notable threat to machine learning systems, given their potential to induce erroneous predictions and classifications. However, within real-world contexts, the essential specifics of the deployed model are frequently treated as a black box, consequently mitigating the vulnerability to such attacks. Thus, enhancing the transferability of the adversarial samples has become a crucial area of research, which heavily relies on selecting appropriate surrogate models. To address this challenge, we propose a novel approach that generates adversarial attacks in a mutual-modality optimization scheme. Our approach is accomplished by leveraging the pre-trained CLIP model. Firstly, we conduct a visual attack on the clean image that causes semantic perturbations on the aligned embedding space with the other textual modality. Then, we apply the corresponding defense on the textual modality by updating the prompts, which forces the re-matching on the perturbed embedding space. Finally, to enhance the attack transferability, we utilize the iterative training strategy on the visual attack and the textual defense, where the two processes optimize from each other. We evaluate our approach on several benchmark datasets and demonstrate that our mutual-modal attack strategy can effectively produce high-transferable attacks, which are stable regardless of the target networks. Our approach outperforms state-of-the-art attack methods and can be readily deployed as a plug-and-play solution.



## **14. Trust, but Verify: Robust Image Segmentation using Deep Learning**

cs.CV

5 Pages, 8 Figures, conference

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2310.16999v3) [paper-pdf](http://arxiv.org/pdf/2310.16999v3)

**Authors**: Fahim Ahmed Zaman, Xiaodong Wu, Weiyu Xu, Milan Sonka, Raghuraman Mudumbai

**Abstract**: We describe a method for verifying the output of a deep neural network for medical image segmentation that is robust to several classes of random as well as worst-case perturbations i.e. adversarial attacks. This method is based on a general approach recently developed by the authors called "Trust, but Verify" wherein an auxiliary verification network produces predictions about certain masked features in the input image using the segmentation as an input. A well-designed auxiliary network will produce high-quality predictions when the input segmentations are accurate, but will produce low-quality predictions when the segmentations are incorrect. Checking the predictions of such a network with the original image allows us to detect bad segmentations. However, to ensure the verification method is truly robust, we need a method for checking the quality of the predictions that does not itself rely on a black-box neural network. Indeed, we show that previous methods for segmentation evaluation that do use deep neural regression networks are vulnerable to false negatives i.e. can inaccurately label bad segmentations as good. We describe the design of a verification network that avoids such vulnerability and present results to demonstrate its robustness compared to previous methods.



## **15. ReRoGCRL: Representation-based Robustness in Goal-Conditioned Reinforcement Learning**

cs.LG

This paper has been accepted in AAAI24  (https://aaai.org/aaai-conference/)

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.07392v3) [paper-pdf](http://arxiv.org/pdf/2312.07392v3)

**Authors**: Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan

**Abstract**: While Goal-Conditioned Reinforcement Learning (GCRL) has gained attention, its algorithmic robustness against adversarial perturbations remains unexplored. The attacks and robust representation training methods that are designed for traditional RL become less effective when applied to GCRL. To address this challenge, we first propose the Semi-Contrastive Representation attack, a novel approach inspired by the adversarial contrastive attack. Unlike existing attacks in RL, it only necessitates information from the policy function and can be seamlessly implemented during deployment. Then, to mitigate the vulnerability of existing GCRL algorithms, we introduce Adversarial Representation Tactics, which combines Semi-Contrastive Adversarial Augmentation with Sensitivity-Aware Regularizer to improve the adversarial robustness of the underlying RL agent against various types of perturbations. Extensive experiments validate the superior performance of our attack and defence methods across multiple state-of-the-art GCRL algorithms. Our tool ReRoGCRL is available at https://github.com/TrustAI/ReRoGCRL.



## **16. Trust, But Verify: A Survey of Randomized Smoothing Techniques**

cs.LG

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.12608v1) [paper-pdf](http://arxiv.org/pdf/2312.12608v1)

**Authors**: Anupriya Kumari, Devansh Bhardwaj, Sukrit Jindal, Sarthak Gupta

**Abstract**: Machine learning models have demonstrated remarkable success across diverse domains but remain vulnerable to adversarial attacks. Empirical defence mechanisms often fall short, as new attacks constantly emerge, rendering existing defences obsolete. A paradigm shift from empirical defences to certification-based defences has been observed in response. Randomized smoothing has emerged as a promising technique among notable advancements. This study reviews the theoretical foundations, empirical effectiveness, and applications of randomized smoothing in verifying machine learning classifiers. We provide an in-depth exploration of the fundamental concepts underlying randomized smoothing, highlighting its theoretical guarantees in certifying robustness against adversarial perturbations. Additionally, we discuss the challenges of existing methodologies and offer insightful perspectives on potential solutions. This paper is novel in its attempt to systemise the existing knowledge in the context of randomized smoothing.



## **17. You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks**

cs.CR

To be published in Proceedings of the 33rd USENIX Security Symposium  (USENIX Security 2024)

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2311.10197v2) [paper-pdf](http://arxiv.org/pdf/2311.10197v2)

**Authors**: Rafael Uetz, Marco Herzog, Louis Hackländer, Simon Schwarz, Martin Henze

**Abstract**: Cyberattacks have grown into a major risk for organizations, with common consequences being data theft, sabotage, and extortion. Since preventive measures do not suffice to repel attacks, timely detection of successful intruders is crucial to stop them from reaching their final goals. For this purpose, many organizations utilize Security Information and Event Management (SIEM) systems to centrally collect security-related events and scan them for attack indicators using expert-written detection rules. However, as we show by analyzing a set of widespread SIEM detection rules, adversaries can evade almost half of them easily, allowing them to perform common malicious actions within an enterprise network without being detected. To remedy these critical detection blind spots, we propose the idea of adaptive misuse detection, which utilizes machine learning to compare incoming events to SIEM rules on the one hand and known-benign events on the other hand to discover successful evasions. Based on this idea, we present AMIDES, an open-source proof-of-concept adaptive misuse detection system. Using four weeks of SIEM events from a large enterprise network and more than 500 hand-crafted evasions, we show that AMIDES successfully detects a majority of these evasions without any false alerts. In addition, AMIDES eases alert analysis by assessing which rules were evaded. Its computational efficiency qualifies AMIDES for real-world operation and hence enables organizations to significantly reduce detection blind spots with moderate effort.



## **18. Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks**

cs.CV

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2310.06958v2) [paper-pdf](http://arxiv.org/pdf/2310.06958v2)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy Vatolin

**Abstract**: Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. Try our benchmark using pip install robustness-benchmark.



## **19. Tensor Train Decomposition for Adversarial Attacks on Computer Vision Models**

math.NA

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.12556v1) [paper-pdf](http://arxiv.org/pdf/2312.12556v1)

**Authors**: Andrei Chertkov, Ivan Oseledets

**Abstract**: Deep neural networks (DNNs) are widely used today, but they are vulnerable to adversarial attacks. To develop effective methods of defense, it is important to understand the potential weak spots of DNNs. Often attacks are organized taking into account the architecture of models (white-box approach) and based on gradient methods, but for real-world DNNs this approach in most cases is impossible. At the same time, several gradient-free optimization algorithms are used to attack black-box models. However, classical methods are often ineffective in the multidimensional case. To organize black-box attacks for computer vision models, in this work, we propose the use of an optimizer based on the low-rank tensor train (TT) format, which has gained popularity in various practical multidimensional applications in recent years. Combined with the attribution of the target image, which is built by the auxiliary (white-box) model, the TT-based optimization method makes it possible to organize an effective black-box attack by small perturbation of pixels in the target image. The superiority of the proposed approach over three popular baselines is demonstrated for five modern DNNs on the ImageNet dataset.



## **20. Counter-Empirical Attacking based on Adversarial Reinforcement Learning for Time-Relevant Scoring System**

cs.LG

Accepted by TKDE

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2311.05144v2) [paper-pdf](http://arxiv.org/pdf/2311.05144v2)

**Authors**: Xiangguo Sun, Hong Cheng, Hang Dong, Bo Qiao, Si Qin, Qingwei Lin

**Abstract**: Scoring systems are commonly seen for platforms in the era of big data. From credit scoring systems in financial services to membership scores in E-commerce shopping platforms, platform managers use such systems to guide users towards the encouraged activity pattern, and manage resources more effectively and more efficiently thereby. To establish such scoring systems, several "empirical criteria" are firstly determined, followed by dedicated top-down design for each factor of the score, which usually requires enormous effort to adjust and tune the scoring function in the new application scenario. What's worse, many fresh projects usually have no ground-truth or any experience to evaluate a reasonable scoring system, making the designing even harder. To reduce the effort of manual adjustment of the scoring function in every new scoring system, we innovatively study the scoring system from the preset empirical criteria without any ground truth, and propose a novel framework to improve the system from scratch. In this paper, we propose a "counter-empirical attacking" mechanism that can generate "attacking" behavior traces and try to break the empirical rules of the scoring system. Then an adversarial "enhancer" is applied to evaluate the scoring system and find the improvement strategy. By training the adversarial learning problem, a proper scoring function can be learned to be robust to the attacking activity traces that are trying to violate the empirical criteria. Extensive experiments have been conducted on two scoring systems including a shared computing resource platform and a financial credit system. The experimental results have validated the effectiveness of our proposed framework.



## **21. Position Bias Mitigation: A Knowledge-Aware Graph Model for Emotion Cause Extraction**

cs.CL

ACL2021 Main Conference, Oral paper

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2106.03518v3) [paper-pdf](http://arxiv.org/pdf/2106.03518v3)

**Authors**: Hanqi Yan, Lin Gui, Gabriele Pergola, Yulan He

**Abstract**: The Emotion Cause Extraction (ECE)} task aims to identify clauses which contain emotion-evoking information for a particular emotion expressed in text. We observe that a widely-used ECE dataset exhibits a bias that the majority of annotated cause clauses are either directly before their associated emotion clauses or are the emotion clauses themselves. Existing models for ECE tend to explore such relative position information and suffer from the dataset bias. To investigate the degree of reliance of existing ECE models on clause relative positions, we propose a novel strategy to generate adversarial examples in which the relative position information is no longer the indicative feature of cause clauses. We test the performance of existing models on such adversarial examples and observe a significant performance drop. To address the dataset bias, we propose a novel graph-based method to explicitly model the emotion triggering paths by leveraging the commonsense knowledge to enhance the semantic dependencies between a candidate clause and an emotion clause. Experimental results show that our proposed approach performs on par with the existing state-of-the-art methods on the original ECE dataset, and is more robust against adversarial attacks compared to existing models.



## **22. QuanShield: Protecting against Side-Channels Attacks using Self-Destructing Enclaves**

cs.CR

15pages, 5 figures, 5 tables

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.11796v1) [paper-pdf](http://arxiv.org/pdf/2312.11796v1)

**Authors**: Shujie Cui, Haohua Li, Yuanhong Li, Zhi Zhang, Lluís Vilanova, Peter Pietzuch

**Abstract**: Trusted Execution Environments (TEEs) allow user processes to create enclaves that protect security-sensitive computation against access from the OS kernel and the hypervisor. Recent work has shown that TEEs are vulnerable to side-channel attacks that allow an adversary to learn secrets shielded in enclaves. The majority of such attacks trigger exceptions or interrupts to trace the control or data flow of enclave execution.   We propose QuanShield, a system that protects enclaves from side-channel attacks that interrupt enclave execution. The main idea behind QuanShield is to strengthen resource isolation by creating an interrupt-free environment on a dedicated CPU core for running enclaves in which enclaves terminate when interrupts occur. QuanShield avoids interrupts by exploiting the tickless scheduling mode supported by recent OS kernels. QuanShield then uses the save area (SA) of the enclave, which is used by the hardware to support interrupt handling, as a second stack. Through an LLVM-based compiler pass, QuanShield modifies enclave instructions to store/load memory references, such as function frame base addresses, to/from the SA. When an interrupt occurs, the hardware overwrites the data in the SA with CPU state, thus ensuring that enclave execution fails. Our evaluation shows that QuanShield significantly raises the bar for interrupt-based attacks with practical overhead.



## **23. Impartial Games: A Challenge for Reinforcement Learning**

cs.LG

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2205.12787v3) [paper-pdf](http://arxiv.org/pdf/2205.12787v3)

**Authors**: Bei Zhou, Søren Riis

**Abstract**: While AlphaZero-style reinforcement learning (RL) algorithms excel in various board games, in this paper we show that they face challenges on impartial games where players share pieces. We present a concrete example of a game - namely the children's game of Nim - and other impartial games that seem to be a stumbling block for AlphaZero-style and similar self-play reinforcement learning algorithms.   Our work is built on the challenges posed by the intricacies of data distribution on the ability of neural networks to learn parity functions, exacerbated by the noisy labels issue. Our findings are consistent with recent studies showing that AlphaZero-style algorithms are vulnerable to adversarial attacks and adversarial perturbations, showing the difficulty of learning to master the games in all legal states.   We show that Nim can be learned on small boards, but the learning progress of AlphaZero-style algorithms dramatically slows down when the board size increases. Intuitively, the difference between impartial games like Nim and partisan games like Chess and Go can be explained by the fact that if a small part of the board is covered for impartial games it is typically not possible to predict whether the position is won or lost as there is often zero correlation between the visible part of a partly blanked-out position and its correct evaluation. This situation starkly contrasts partisan games where a partly blanked-out board position typically provides abundant or at least non-trifle information about the value of the fully uncovered position.



## **24. The Ultimate Combo: Boosting Adversarial Example Transferability by Composing Data Augmentations**

cs.CV

18 pages

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11309v1) [paper-pdf](http://arxiv.org/pdf/2312.11309v1)

**Authors**: Zebin Yun, Achi-Or Weingarten, Eyal Ronen, Mahmood Sharif

**Abstract**: Transferring adversarial examples (AEs) from surrogate machine-learning (ML) models to target models is commonly used in black-box adversarial robustness evaluation. Attacks leveraging certain data augmentation, such as random resizing, have been found to help AEs generalize from surrogates to targets. Yet, prior work has explored limited augmentations and their composition. To fill the gap, we systematically studied how data augmentation affects transferability. Particularly, we explored 46 augmentation techniques of seven categories originally proposed to help ML models generalize to unseen benign samples, and assessed how they impact transferability, when applied individually or composed. Performing exhaustive search on a small subset of augmentation techniques and genetic search on all techniques, we identified augmentation combinations that can help promote transferability. Extensive experiments with the ImageNet and CIFAR-10 datasets and 18 models showed that simple color-space augmentations (e.g., color to greyscale) outperform the state of the art when combined with standard augmentations, such as translation and scaling. Additionally, we discovered that composing augmentations impacts transferability mostly monotonically (i.e., more methods composed $\rightarrow$ $\ge$ transferability). We also found that the best composition significantly outperformed the state of the art (e.g., 93.7% vs. $\le$ 82.7% average transferability on ImageNet from normally trained surrogates to adversarially trained targets). Lastly, our theoretical analysis, backed up by empirical evidence, intuitively explain why certain augmentations help improve transferability.



## **25. Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model**

cs.CV

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11285v1) [paper-pdf](http://arxiv.org/pdf/2312.11285v1)

**Authors**: Decheng Liu, Xijun Wang, Chunlei Peng, Nannan Wang, Ruiming Hu, Xinbo Gao

**Abstract**: Adversarial attacks involve adding perturbations to the source image to cause misclassification by the target model, which demonstrates the potential of attacking face recognition models. Existing adversarial face image generation methods still can't achieve satisfactory performance because of low transferability and high detectability. In this paper, we propose a unified framework Adv-Diffusion that can generate imperceptible adversarial identity perturbations in the latent space but not the raw pixel space, which utilizes strong inpainting capabilities of the latent diffusion model to generate realistic adversarial images. Specifically, we propose the identity-sensitive conditioned diffusion generative model to generate semantic perturbations in the surroundings. The designed adaptive strength-based adversarial perturbation algorithm can ensure both attack transferability and stealthiness. Extensive qualitative and quantitative experiments on the public FFHQ and CelebA-HQ datasets prove the proposed method achieves superior performance compared with the state-of-the-art methods without an extra generative model training process. The source code is available at https://github.com/kopper-xdu/Adv-Diffusion.



## **26. Protect Your Score: Contact Tracing With Differential Privacy Guarantees**

cs.CR

Accepted to The 38th Annual AAAI Conference on Artificial  Intelligence (AAAI 2024)

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11581v1) [paper-pdf](http://arxiv.org/pdf/2312.11581v1)

**Authors**: Rob Romijnders, Christos Louizos, Yuki M. Asano, Max Welling

**Abstract**: The pandemic in 2020 and 2021 had enormous economic and societal consequences, and studies show that contact tracing algorithms can be key in the early containment of the virus. While large strides have been made towards more effective contact tracing algorithms, we argue that privacy concerns currently hold deployment back. The essence of a contact tracing algorithm constitutes the communication of a risk score. Yet, it is precisely the communication and release of this score to a user that an adversary can leverage to gauge the private health status of an individual. We pinpoint a realistic attack scenario and propose a contact tracing algorithm with differential privacy guarantees against this attack. The algorithm is tested on the two most widely used agent-based COVID19 simulators and demonstrates superior performance in a wide range of settings. Especially for realistic test scenarios and while releasing each risk score with epsilon=1 differential privacy, we achieve a two to ten-fold reduction in the infection rate of the virus. To the best of our knowledge, this presents the first contact tracing algorithm with differential privacy guarantees when revealing risk scores for COVID19.



## **27. A Survey of Side-Channel Attacks in Context of Cache -- Taxonomies, Analysis and Mitigation**

cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11094v1) [paper-pdf](http://arxiv.org/pdf/2312.11094v1)

**Authors**: Ankit Pulkit, Smita Naval, Vijay Laxmi

**Abstract**: Side-channel attacks have become prominent attack surfaces in cyberspace. Attackers use the side information generated by the system while performing a task. Among the various side-channel attacks, cache side-channel attacks are leading as there has been an enormous growth in cache memory size in last decade, especially Last Level Cache (LLC). The adversary infers the information from the observable behavior of shared cache memory. This paper covers the detailed study of cache side-channel attacks and compares different microarchitectures in the context of side-channel attacks. Our main contributions are: (1) We have summarized the fundamentals and essentials of side-channel attacks and various attack surfaces (taxonomies). We also discussed different exploitation techniques, highlighting their capabilities and limitations. (2) We discussed cache side-channel attacks and analyzed the existing literature on cache side-channel attacks on various parameters like microarchitectures, cross-core exploitation, methodology, target, etc. (3) We discussed the detailed analysis of the existing mitigation strategies to prevent cache side-channel attacks. The analysis includes hardware- and software-based countermeasures, examining their strengths and weaknesses. We also discussed the challenges and trade-offs associated with mitigation strategies. This survey is supposed to provide a deeper understanding of the threats posed by these attacks to the research community with valuable insights into effective defense mechanisms.



## **28. Model Stealing Attack against Recommender System**

cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11571v1) [paper-pdf](http://arxiv.org/pdf/2312.11571v1)

**Authors**: Zhihao Zhu, Rui Fan, Chenwang Wu, Yi Yang, Defu Lian, Enhong Chen

**Abstract**: Recent studies have demonstrated the vulnerability of recommender systems to data privacy attacks. However, research on the threat to model privacy in recommender systems, such as model stealing attacks, is still in its infancy. Some adversarial attacks have achieved model stealing attacks against recommender systems, to some extent, by collecting abundant training data of the target model (target data) or making a mass of queries. In this paper, we constrain the volume of available target data and queries and utilize auxiliary data, which shares the item set with the target data, to promote model stealing attacks. Although the target model treats target and auxiliary data differently, their similar behavior patterns allow them to be fused using an attention mechanism to assist attacks. Besides, we design stealing functions to effectively extract the recommendation list obtained by querying the target model. Experimental results show that the proposed methods are applicable to most recommender systems and various scenarios and exhibit excellent attack performance on multiple datasets.



## **29. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023; (v2:  fixed typos)

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.08793v2) [paper-pdf](http://arxiv.org/pdf/2312.08793v2)

**Authors**: Tony T. Wang, Miles Wang, Kaivalya Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .



## **30. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.09494v2) [paper-pdf](http://arxiv.org/pdf/2312.09494v2)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.



## **31. Security Defense of Large Scale Networks Under False Data Injection Attacks: An Attack Detection Scheduling Approach**

eess.SY

14 pages, 13 figures

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2212.05500v4) [paper-pdf](http://arxiv.org/pdf/2212.05500v4)

**Authors**: Yuhan Suo, Senchun Chai, Runqi Chai, Zhong-Hua Pang, Yuanqing Xia, Guo-Ping Liu

**Abstract**: In large-scale networks, communication links between nodes are easily injected with false data by adversaries. This paper proposes a novel security defense strategy from the perspective of attack detection scheduling to ensure the security of the network. Based on the proposed strategy, each sensor can directly exclude suspicious sensors from its neighboring set. First, the problem of selecting suspicious sensors is formulated as a combinatorial optimization problem, which is non-deterministic polynomial-time hard (NP-hard). To solve this problem, the original function is transformed into a submodular function. Then, we propose an attack detection scheduling algorithm based on the sequential submodular optimization theory, which incorporates \emph{expert problem} to better utilize historical information to guide the sensor selection task at the current moment. For different attack strategies, theoretical results show that the average optimization rate of the proposed algorithm has a lower bound, and the error expectation is bounded. In addition, under two kinds of insecurity conditions, the proposed algorithm can guarantee the security of the entire network from the perspective of the augmented estimation error. Finally, the effectiveness of the developed method is verified by the numerical simulation and practical experiment.



## **32. Security for Machine Learning-based Software Systems: a survey of threats, practices and challenges**

cs.CR

Accepted at ACM Computing Surveys

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2201.04736v2) [paper-pdf](http://arxiv.org/pdf/2201.04736v2)

**Authors**: Huaming Chen, M. Ali Babar

**Abstract**: The rapid development of Machine Learning (ML) has demonstrated superior performance in many areas, such as computer vision, video and speech recognition. It has now been increasingly leveraged in software systems to automate the core tasks. However, how to securely develop the machine learning-based modern software systems (MLBSS) remains a big challenge, for which the insufficient consideration will largely limit its application in safety-critical domains. One concern is that the present MLBSS development tends to be rush, and the latent vulnerabilities and privacy issues exposed to external users and attackers will be largely neglected and hard to be identified. Additionally, machine learning-based software systems exhibit different liabilities towards novel vulnerabilities at different development stages from requirement analysis to system maintenance, due to its inherent limitations from the model and data and the external adversary capabilities. The successful generation of such intelligent systems will thus solicit dedicated efforts jointly from different research areas, i.e., software engineering, system security and machine learning. Most of the recent works regarding the security issues for ML have a strong focus on the data and models, which has brought adversarial attacks into consideration. In this work, we consider that security for machine learning-based software systems may arise from inherent system defects or external adversarial attacks, and the secure development practices should be taken throughout the whole lifecycle. While machine learning has become a new threat domain for existing software engineering practices, there is no such review work covering the topic. Overall, we present a holistic review regarding the security for MLBSS, which covers a systematic understanding from a structure review of three distinct aspects in terms of security threats...



## **33. Single Image Backdoor Inversion via Robust Smoothed Classifiers**

cs.CV

CVPR 2023. v2: improved writing

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2303.00215v2) [paper-pdf](http://arxiv.org/pdf/2303.00215v2)

**Authors**: Mingjie Sun, J. Zico Kolter

**Abstract**: Backdoor inversion, a central step in many backdoor defenses, is a reverse-engineering process to recover the hidden backdoor trigger inserted into a machine learning model. Existing approaches tackle this problem by searching for a backdoor pattern that is able to flip a set of clean images into the target class, while the exact size needed of this support set is rarely investigated. In this work, we present a new approach for backdoor inversion, which is able to recover the hidden backdoor with as few as a single image. Insipired by recent advances in adversarial robustness, our method SmoothInv starts from a single clean image, and then performs projected gradient descent towards the target class on a robust smoothed version of the original backdoored classifier. We find that backdoor patterns emerge naturally from such optimization process. Compared to existing backdoor inversion methods, SmoothInv introduces minimum optimization variables and does not require complex regularization schemes. We perform a comprehensive quantitative and qualitative study on backdoored classifiers obtained from existing backdoor attacks. We demonstrate that SmoothInv consistently recovers successful backdoors from single images: for backdoored ImageNet classifiers, our reconstructed backdoors have close to 100% attack success rates. We also show that they maintain high fidelity to the underlying true backdoors. Last, we propose and analyze two countermeasures to our approach and show that SmoothInv remains robust in the face of an adaptive attacker. Our code is available at https://github.com/locuslab/smoothinv.



## **34. Synthesizing Black-box Anti-forensics DeepFakes with High Visual Quality**

cs.CV

Accepted for publication at ICASSP 2024

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2312.10713v1) [paper-pdf](http://arxiv.org/pdf/2312.10713v1)

**Authors**: Bing Fan, Shu Hu, Feng Ding

**Abstract**: DeepFake, an AI technology for creating facial forgeries, has garnered global attention. Amid such circumstances, forensics researchers focus on developing defensive algorithms to counter these threats. In contrast, there are techniques developed for enhancing the aggressiveness of DeepFake, e.g., through anti-forensics attacks, to disrupt forensic detectors. However, such attacks often sacrifice image visual quality for improved undetectability. To address this issue, we propose a method to generate novel adversarial sharpening masks for launching black-box anti-forensics attacks. Unlike many existing arts, with such perturbations injected, DeepFakes could achieve high anti-forensics performance while exhibiting pleasant sharpening visual effects. After experimental evaluations, we prove that the proposed method could successfully disrupt the state-of-the-art DeepFake detectors. Besides, compared with the images processed by existing DeepFake anti-forensics methods, the visual qualities of anti-forensics DeepFakes rendered by the proposed method are significantly refined.



## **35. Analisis Eksploratif Dan Augmentasi Data NSL-KDD Menggunakan Deep Generative Adversarial Networks Untuk Meningkatkan Performa Algoritma Extreme Gradient Boosting Dalam Klasifikasi Jenis Serangan Siber**

cs.CR

in Indonesian language

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2312.10669v1) [paper-pdf](http://arxiv.org/pdf/2312.10669v1)

**Authors**: K. P. Santoso, F. A. Madany, H. Suryotrisongko

**Abstract**: This study proposes the implementation of Deep Generative Adversarial Networks (GANs) for augmenting the NSL-KDD dataset. The primary objective is to enhance the efficacy of eXtreme Gradient Boosting (XGBoost) in the classification of cyber-attacks on the NSL-KDD dataset. As a result, the method proposed in this research achieved an accuracy of 99.53% using the XGBoost model without data augmentation with GAN, and 99.78% with data augmentation using GAN.



## **36. UltraClean: A Simple Framework to Train Robust Neural Networks against Backdoor Attacks**

cs.CR

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2312.10657v1) [paper-pdf](http://arxiv.org/pdf/2312.10657v1)

**Authors**: Bingyin Zhao, Yingjie Lao

**Abstract**: Backdoor attacks are emerging threats to deep neural networks, which typically embed malicious behaviors into a victim model by injecting poisoned samples. Adversaries can activate the injected backdoor during inference by presenting the trigger on input images. Prior defensive methods have achieved remarkable success in countering dirty-label backdoor attacks where the labels of poisoned samples are often mislabeled. However, these approaches do not work for a recent new type of backdoor -- clean-label backdoor attacks that imperceptibly modify poisoned data and hold consistent labels. More complex and powerful algorithms are demanded to defend against such stealthy attacks. In this paper, we propose UltraClean, a general framework that simplifies the identification of poisoned samples and defends against both dirty-label and clean-label backdoor attacks. Given the fact that backdoor triggers introduce adversarial noise that intensifies in feed-forward propagation, UltraClean first generates two variants of training samples using off-the-shelf denoising functions. It then measures the susceptibility of training samples leveraging the error amplification effect in DNNs, which dilates the noise difference between the original image and denoised variants. Lastly, it filters out poisoned samples based on the susceptibility to thwart the backdoor implantation. Despite its simplicity, UltraClean achieves a superior detection rate across various datasets and significantly reduces the backdoor attack success rate while maintaining a decent model accuracy on clean data, outperforming existing defensive methods by a large margin. Code is available at https://github.com/bxz9200/UltraClean.



## **37. Rethinking Robustness of Model Attributions**

cs.LG

Accepted AAAI 2024

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.10534v1) [paper-pdf](http://arxiv.org/pdf/2312.10534v1)

**Authors**: Sandesh Kamath, Sankalp Mittal, Amit Deshpande, Vineeth N Balasubramanian

**Abstract**: For machine learning models to be reliable and trustworthy, their decisions must be interpretable. As these models find increasing use in safety-critical applications, it is important that not just the model predictions but also their explanations (as feature attributions) be robust to small human-imperceptible input perturbations. Recent works have shown that many attribution methods are fragile and have proposed improvements in either these methods or the model training. We observe two main causes for fragile attributions: first, the existing metrics of robustness (e.g., top-k intersection) over-penalize even reasonable local shifts in attribution, thereby making random perturbations to appear as a strong attack, and second, the attribution can be concentrated in a small region even when there are multiple important parts in an image. To rectify this, we propose simple ways to strengthen existing metrics and attribution methods that incorporate locality of pixels in robustness metrics and diversity of pixel locations in attributions. Towards the role of model training in attributional robustness, we empirically observe that adversarially trained models have more robust attributions on smaller datasets, however, this advantage disappears in larger datasets. Code is available at https://github.com/ksandeshk/LENS.



## **38. Transformers in Unsupervised Structure-from-Motion**

cs.CV

International Joint Conference on Computer Vision, Imaging and  Computer Graphics. Cham: Springer Nature Switzerland, 2022. Published at  "Communications in Computer and Information Science, vol 1815. Springer  Nature". arXiv admin note: text overlap with arXiv:2202.03131

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.10529v1) [paper-pdf](http://arxiv.org/pdf/2312.10529v1)

**Authors**: Hemang Chawla, Arnav Varma, Elahe Arani, Bahram Zonooz

**Abstract**: Transformers have revolutionized deep learning based computer vision with improved performance as well as robustness to natural corruptions and adversarial attacks. Transformers are used predominantly for 2D vision tasks, including image classification, semantic segmentation, and object detection. However, robots and advanced driver assistance systems also require 3D scene understanding for decision making by extracting structure-from-motion (SfM). We propose a robust transformer-based monocular SfM method that learns to predict monocular pixel-wise depth, ego vehicle's translation and rotation, as well as camera's focal length and principal point, simultaneously. With experiments on KITTI and DDAD datasets, we demonstrate how to adapt different vision transformers and compare them against contemporary CNN-based methods. Our study shows that transformer-based architecture, though lower in run-time efficiency, achieves comparable performance while being more robust against natural corruptions, as well as untargeted and targeted attacks.



## **39. IoTGAN: GAN Powered Camouflage Against Machine Learning Based IoT Device Identification**

cs.CR

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2201.03281v2) [paper-pdf](http://arxiv.org/pdf/2201.03281v2)

**Authors**: Tao Hou, Tao Wang, Zhuo Lu, Yao Liu, Yalin Sagduyu

**Abstract**: With the proliferation of IoT devices, researchers have developed a variety of IoT device identification methods with the assistance of machine learning. Nevertheless, the security of these identification methods mostly depends on collected training data. In this research, we propose a novel attack strategy named IoTGAN to manipulate an IoT device's traffic such that it can evade machine learning based IoT device identification. In the development of IoTGAN, we have two major technical challenges: (i) How to obtain the discriminative model in a black-box setting, and (ii) How to add perturbations to IoT traffic through the manipulative model, so as to evade the identification while not influencing the functionality of IoT devices. To address these challenges, a neural network based substitute model is used to fit the target model in black-box settings, it works as a discriminative model in IoTGAN. A manipulative model is trained to add adversarial perturbations into the IoT device's traffic to evade the substitute model. Experimental results show that IoTGAN can successfully achieve the attack goals. We also develop efficient countermeasures to protect machine learning based IoT device identification from been undermined by IoTGAN.



## **40. Robust Communicative Multi-Agent Reinforcement Learning with Active Defense**

cs.MA

Accepted by AAAI 2024

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.11545v1) [paper-pdf](http://arxiv.org/pdf/2312.11545v1)

**Authors**: Lebin Yu, Yunbo Qiu, Quanming Yao, Yuan Shen, Xudong Zhang, Jian Wang

**Abstract**: Communication in multi-agent reinforcement learning (MARL) has been proven to effectively promote cooperation among agents recently. Since communication in real-world scenarios is vulnerable to noises and adversarial attacks, it is crucial to develop robust communicative MARL technique. However, existing research in this domain has predominantly focused on passive defense strategies, where agents receive all messages equally, making it hard to balance performance and robustness. We propose an active defense strategy, where agents automatically reduce the impact of potentially harmful messages on the final decision. There are two challenges to implement this strategy, that are defining unreliable messages and adjusting the unreliable messages' impact on the final decision properly. To address them, we design an Active Defense Multi-Agent Communication framework (ADMAC), which estimates the reliability of received messages and adjusts their impact on the final decision accordingly with the help of a decomposable decision structure. The superiority of ADMAC over existing methods is validated by experiments in three communication-critical tasks under four types of attacks.



## **41. PPIDSG: A Privacy-Preserving Image Distribution Sharing Scheme with GAN in Federated Learning**

cs.LG

Accepted by AAAI 2024

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.10380v1) [paper-pdf](http://arxiv.org/pdf/2312.10380v1)

**Authors**: Yuting Ma, Yuanzhi Yao, Xiaohua Xu

**Abstract**: Federated learning (FL) has attracted growing attention since it allows for privacy-preserving collaborative training on decentralized clients without explicitly uploading sensitive data to the central server. However, recent works have revealed that it still has the risk of exposing private data to adversaries. In this paper, we conduct reconstruction attacks and enhance inference attacks on various datasets to better understand that sharing trained classification model parameters to a central server is the main problem of privacy leakage in FL. To tackle this problem, a privacy-preserving image distribution sharing scheme with GAN (PPIDSG) is proposed, which consists of a block scrambling-based encryption algorithm, an image distribution sharing method, and local classification training. Specifically, our method can capture the distribution of a target image domain which is transformed by the block encryption algorithm, and upload generator parameters to avoid classifier sharing with negligible influence on model performance. Furthermore, we apply a feature extractor to motivate model utility and train it separately from the classifier. The extensive experimental results and security analyses demonstrate the superiority of our proposed scheme compared to other state-of-the-art defense methods. The code is available at https://github.com/ytingma/PPIDSG.



## **42. Perturbation-Invariant Adversarial Training for Neural Ranking Models: Improving the Effectiveness-Robustness Trade-Off**

cs.IR

Accepted by AAAI 24

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.10329v1) [paper-pdf](http://arxiv.org/pdf/2312.10329v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Mingkun Zhang, Wei Chen, Maarten de Rijke, Jiafeng Guo, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have shown great success in information retrieval (IR). But their predictions can easily be manipulated using adversarial examples, which are crafted by adding imperceptible perturbations to legitimate documents. This vulnerability raises significant concerns about their reliability and hinders the widespread deployment of NRMs. By incorporating adversarial examples into training data, adversarial training has become the de facto defense approach to adversarial attacks against NRMs. However, this defense mechanism is subject to a trade-off between effectiveness and adversarial robustness. In this study, we establish theoretical guarantees regarding the effectiveness-robustness trade-off in NRMs. We decompose the robust ranking error into two components, i.e., a natural ranking error for effectiveness evaluation and a boundary ranking error for assessing adversarial robustness. Then, we define the perturbation invariance of a ranking model and prove it to be a differentiable upper bound on the boundary ranking error for attainable computation. Informed by our theoretical analysis, we design a novel \emph{perturbation-invariant adversarial training} (PIAT) method for ranking models to achieve a better effectiveness-robustness trade-off. We design a regularized surrogate loss, in which one term encourages the effectiveness to be maximized while the regularization term encourages the output to be smooth, so as to improve adversarial robustness. Experimental results on several ranking models demonstrate the superiority of PITA compared to existing adversarial defenses.



## **43. Enhancing Accuracy and Robustness of Steering Angle Prediction with Attention Mechanism**

cs.CV

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2211.11133v3) [paper-pdf](http://arxiv.org/pdf/2211.11133v3)

**Authors**: Swetha Nadella, Pramiti Barua, Jeremy C. Hagler, David J. Lamb, Qing Tian

**Abstract**: In this paper, we investigate the two most popular families of deep neural architectures (i.e., ResNets and InceptionNets) for the autonomous driving task of steering angle prediction. To ensure a comprehensive comparison, we conducted experiments on the Kaggle SAP dataset and custom dataset and carefully examined a range of different model sizes within both the ResNet and InceptionNet families. Our derived models can achieve state-of-the-art results in terms of steering angle MSE. In addition to this analysis, we introduced the attention mechanism to enhance steering angle prediction. This attention mechanism facilitated an in-depth exploration of the model's selective focus on essential elements within the input data. Furthermore, recognizing the importance of security and robustness in autonomous driving assessed the resilience of our models to adversarial attacks.



## **44. Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models**

cs.LG

19 pages, 6 figures; AAAI24

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2305.12066v2) [paper-pdf](http://arxiv.org/pdf/2305.12066v2)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-task learning (MTL) creates a single machine learning model called multi-task model to simultaneously perform multiple tasks. Although the security of single task classifiers has been extensively studied, there are several critical security research questions for multi-task models including 1) How secure are multi-task models to single task adversarial machine learning attacks, 2) Can adversarial attacks be designed to attack multiple tasks simultaneously, and 3) Does task sharing and adversarial training increase multi-task model robustness to adversarial attacks? In this paper, we answer these questions through careful analysis and rigorous experimentation. First, we develop na\"ive adaptation of single-task white-box attacks and analyze their inherent drawbacks. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking a multi-task model as an optimization problem based on averaged relative loss change, which can be solved by approximating the problem as an integer linear programming problem. Extensive evaluation on two popular MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrates the effectiveness of DGBA compared to na\"ive multi-task attack baselines on both clean and adversarially trained multi-task models. The results also reveal a fundamental trade-off between improving task accuracy by sharing parameters across tasks and undermining model robustness due to increased attack transferability from parameter sharing. DGBA is open-sourced and available at https://github.com/zhanglijun95/MTLAttack-DGBA.



## **45. Exploring Adversarial Robustness of Vision Transformers in the Spectral Perspective**

cs.CV

Accepted in IEEE/CVF Winter Conference on Applications of Computer  Vision (WACV) 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2208.09602v2) [paper-pdf](http://arxiv.org/pdf/2208.09602v2)

**Authors**: Gihyun Kim, Juyeop Kim, Jong-Seok Lee

**Abstract**: The Vision Transformer has emerged as a powerful tool for image classification tasks, surpassing the performance of convolutional neural networks (CNNs). Recently, many researchers have attempted to understand the robustness of Transformers against adversarial attacks. However, previous researches have focused solely on perturbations in the spatial domain. This paper proposes an additional perspective that explores the adversarial robustness of Transformers against frequency-selective perturbations in the spectral domain. To facilitate comparison between these two domains, an attack framework is formulated as a flexible tool for implementing attacks on images in the spatial and spectral domains. The experiments reveal that Transformers rely more on phase and low frequency information, which can render them more vulnerable to frequency-selective attacks than CNNs. This work offers new insights into the properties and adversarial robustness of Transformers.



## **46. LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer**

cs.CV

13 pages, 3 figures. Accepted to AAAI 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09935v1) [paper-pdf](http://arxiv.org/pdf/2312.09935v1)

**Authors**: Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu

**Abstract**: Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.



## **47. A Game-theoretic Framework for Privacy-preserving Federated Learning**

cs.LG

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2304.05836v2) [paper-pdf](http://arxiv.org/pdf/2304.05836v2)

**Authors**: Xiaojin Zhang, Lixin Fan, Siwei Wang, Wenjie Li, Kai Chen, Qiang Yang

**Abstract**: In federated learning, benign participants aim to optimize a global model collaboratively. However, the risk of \textit{privacy leakage} cannot be ignored in the presence of \textit{semi-honest} adversaries. Existing research has focused either on designing protection mechanisms or on inventing attacking mechanisms. While the battle between defenders and attackers seems never-ending, we are concerned with one critical question: is it possible to prevent potential attacks in advance? To address this, we propose the first game-theoretic framework that considers both FL defenders and attackers in terms of their respective payoffs, which include computational costs, FL model utilities, and privacy leakage risks. We name this game the federated learning privacy game (FLPG), in which neither defenders nor attackers are aware of all participants' payoffs.   To handle the \textit{incomplete information} inherent in this situation, we propose associating the FLPG with an \textit{oracle} that has two primary responsibilities. First, the oracle provides lower and upper bounds of the payoffs for the players. Second, the oracle acts as a correlation device, privately providing suggested actions to each player. With this novel framework, we analyze the optimal strategies of defenders and attackers. Furthermore, we derive and demonstrate conditions under which the attacker, as a rational decision-maker, should always follow the oracle's suggestion \textit{not to attack}.



## **48. Categorical composable cryptography: extended version**

cs.CR

Extended version of arXiv:2105.05949 which appeared in FoSSaCS 2022

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2208.13232v4) [paper-pdf](http://arxiv.org/pdf/2208.13232v4)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstract**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. We conclude by using string diagrams to rederive the security of the one-time pad, correctness of Diffie-Hellman key exchange and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting. On the way, we exhibit two categorical constructions of resource theories that might be of independent interest: one capturing resources shared among multiple parties and one capturing resource conversions that succeed asymptotically.



## **49. FlowMur: A Stealthy and Practical Audio Backdoor Attack with Limited Knowledge**

cs.CR

To appear at lEEE Symposium on Security & Privacy (Oakland) 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09665v1) [paper-pdf](http://arxiv.org/pdf/2312.09665v1)

**Authors**: Jiahe Lan, Jie Wang, Baochen Yan, Zheng Yan, Elisa Bertino

**Abstract**: Speech recognition systems driven by DNNs have revolutionized human-computer interaction through voice interfaces, which significantly facilitate our daily lives. However, the growing popularity of these systems also raises special concerns on their security, particularly regarding backdoor attacks. A backdoor attack inserts one or more hidden backdoors into a DNN model during its training process, such that it does not affect the model's performance on benign inputs, but forces the model to produce an adversary-desired output if a specific trigger is present in the model input. Despite the initial success of current audio backdoor attacks, they suffer from the following limitations: (i) Most of them require sufficient knowledge, which limits their widespread adoption. (ii) They are not stealthy enough, thus easy to be detected by humans. (iii) Most of them cannot attack live speech, reducing their practicality. To address these problems, in this paper, we propose FlowMur, a stealthy and practical audio backdoor attack that can be launched with limited knowledge. FlowMur constructs an auxiliary dataset and a surrogate model to augment adversary knowledge. To achieve dynamicity, it formulates trigger generation as an optimization problem and optimizes the trigger over different attachment positions. To enhance stealthiness, we propose an adaptive data poisoning method according to Signal-to-Noise Ratio (SNR). Furthermore, ambient noise is incorporated into the process of trigger generation and data poisoning to make FlowMur robust to ambient noise and improve its practicality. Extensive experiments conducted on two datasets demonstrate that FlowMur achieves high attack performance in both digital and physical settings while remaining resilient to state-of-the-art defenses. In particular, a human study confirms that triggers generated by FlowMur are not easily detected by participants.



## **50. Unsupervised and Supervised learning by Dense Associative Memory under replica symmetry breaking**

cond-mat.dis-nn

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09638v1) [paper-pdf](http://arxiv.org/pdf/2312.09638v1)

**Authors**: Linda Albanese, Andrea Alessandrelli, Alessia Annibale, Adriano Barra

**Abstract**: Statistical mechanics of spin glasses is one of the main strands toward a comprehension of information processing by neural networks and learning machines. Tackling this approach, at the fairly standard replica symmetric level of description, recently Hebbian attractor networks with multi-node interactions (often called Dense Associative Memories) have been shown to outperform their classical pairwise counterparts in a number of tasks, from their robustness against adversarial attacks and their capability to work with prohibitively weak signals to their supra-linear storage capacities. Focusing on mathematical techniques more than computational aspects, in this paper we relax the replica symmetric assumption and we derive the one-step broken-replica-symmetry picture of supervised and unsupervised learning protocols for these Dense Associative Memories: a phase diagram in the space of the control parameters is achieved, independently, both via the Parisi's hierarchy within then replica trick as well as via the Guerra's telescope within the broken-replica interpolation. Further, an explicit analytical investigation is provided to deepen both the big-data and ground state limits of these networks as well as a proof that replica symmetry breaking does not alter the thresholds for learning and slightly increases the maximal storage capacity. Finally the De Almeida and Thouless line, depicting the onset of instability of a replica symmetric description, is also analytically derived highlighting how, crossed this boundary, the broken replica description should be preferred.



