# Latest Adversarial Attack Papers
**update at 2024-04-03 21:09:48**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02151v1) [paper-pdf](http://arxiv.org/pdf/2404.02151v1)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). We provide the code, prompts, and logs of the attacks at https://github.com/tml-epfl/llm-adaptive-attacks.



## **2. Red-Teaming Segment Anything Model**

cs.CV

CVPR 2024 - The 4th Workshop of Adversarial Machine Learning on  Computer Vision: Robustness of Foundation Models

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02067v1) [paper-pdf](http://arxiv.org/pdf/2404.02067v1)

**Authors**: Krzysztof Jankowski, Bartlomiej Sobieski, Mateusz Kwiatkowski, Jakub Szulc, Michal Janik, Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Foundation models have emerged as pivotal tools, tackling many complex tasks through pre-training on vast datasets and subsequent fine-tuning for specific applications. The Segment Anything Model is one of the first and most well-known foundation models for computer vision segmentation tasks. This work presents a multi-faceted red-teaming analysis that tests the Segment Anything Model against challenging tasks: (1) We analyze the impact of style transfer on segmentation masks, demonstrating that applying adverse weather conditions and raindrops to dashboard images of city roads significantly distorts generated masks. (2) We focus on assessing whether the model can be used for attacks on privacy, such as recognizing celebrities' faces, and show that the model possesses some undesired knowledge in this task. (3) Finally, we check how robust the model is to adversarial attacks on segmentation masks under text prompts. We not only show the effectiveness of popular white-box attacks and resistance to black-box attacks but also introduce a novel approach - Focused Iterative Gradient Attack (FIGA) that combines white-box approaches to construct an efficient attack resulting in a smaller number of modified pixels. All of our testing methods and analyses indicate a need for enhanced safety measures in foundation models for image segmentation.



## **3. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

cs.CL

NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2311.09447v2) [paper-pdf](http://arxiv.org/pdf/2311.09447v2)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an adversarial assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose advCoU, an extended Chain of Utterances-based (CoU) prompting strategy by incorporating carefully crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.



## **4. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16591v3) [paper-pdf](http://arxiv.org/pdf/2403.16591v3)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between LDP and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. The relationship between LDP and Maximum Bayesian Privacy (MBP) is first revealed, demonstrating that under uniform prior distribution, a mechanism satisfying $\xi$-LDP will satisfy $\xi$-MBP and conversely $\xi$-MBP also confers 2$\xi$-LDP. Our next theoretical contribution are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms. Our work not only lays the groundwork for future empirical exploration but also promises to facilitate the design of privacy-preserving algorithms, thereby fostering the development of trustworthy machine learning solutions.



## **5. PatchCURE: Improving Certifiable Robustness, Model Utility, and Computation Efficiency of Adversarial Patch Defenses**

cs.CV

USENIX Security 2024. (extended) technical report

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2310.13076v2) [paper-pdf](http://arxiv.org/pdf/2310.13076v2)

**Authors**: Chong Xiang, Tong Wu, Sihui Dai, Jonathan Petit, Suman Jana, Prateek Mittal

**Abstract**: State-of-the-art defenses against adversarial patch attacks can now achieve strong certifiable robustness with a marginal drop in model utility. However, this impressive performance typically comes at the cost of 10-100x more inference-time computation compared to undefended models -- the research community has witnessed an intense three-way trade-off between certifiable robustness, model utility, and computation efficiency. In this paper, we propose a defense framework named PatchCURE to approach this trade-off problem. PatchCURE provides sufficient "knobs" for tuning defense performance and allows us to build a family of defenses: the most robust PatchCURE instance can match the performance of any existing state-of-the-art defense (without efficiency considerations); the most efficient PatchCURE instance has similar inference efficiency as undefended models. Notably, PatchCURE achieves state-of-the-art robustness and utility performance across all different efficiency levels, e.g., 16-23% absolute clean accuracy and certified robust accuracy advantages over prior defenses when requiring computation efficiency to be close to undefended models. The family of PatchCURE defenses enables us to flexibly choose appropriate defenses to satisfy given computation and/or utility constraints in practice.



## **6. Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack**

cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01907v1) [paper-pdf](http://arxiv.org/pdf/2404.01907v1)

**Authors**: Ying Zhou, Ben He, Le Sun

**Abstract**: With the development of large language models (LLMs), detecting whether text is generated by a machine becomes increasingly challenging in the face of malicious use cases like the spread of false information, protection of intellectual property, and prevention of academic plagiarism. While well-trained text detectors have demonstrated promising performance on unseen test data, recent research suggests that these detectors have vulnerabilities when dealing with adversarial attacks such as paraphrasing. In this paper, we propose a framework for a broader class of adversarial attacks, designed to perform minor perturbations in machine-generated content to evade detection. We consider two attack settings: white-box and black-box, and employ adversarial learning in dynamic scenarios to assess the potential enhancement of the current detection model's robustness against such attacks. The empirical results reveal that the current detection models can be compromised in as little as 10 seconds, leading to the misclassification of machine-generated text as human-written content. Furthermore, we explore the prospect of improving the model's robustness over iterative adversarial learning. Although some improvements in model robustness are observed, practical applications still face significant challenges. These findings shed light on the future development of AI-text detectors, emphasizing the need for more accurate and robust detection methods.



## **7. Defense without Forgetting: Continual Adversarial Defense with Anisotropic & Isotropic Pseudo Replay**

cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01828v1) [paper-pdf](http://arxiv.org/pdf/2404.01828v1)

**Authors**: Yuhang Zhou, Zhongyun Hua

**Abstract**: Deep neural networks have demonstrated susceptibility to adversarial attacks. Adversarial defense techniques often focus on one-shot setting to maintain robustness against attack. However, new attacks can emerge in sequences in real-world deployment scenarios. As a result, it is crucial for a defense model to constantly adapt to new attacks, but the adaptation process can lead to catastrophic forgetting of previously defended against attacks. In this paper, we discuss for the first time the concept of continual adversarial defense under a sequence of attacks, and propose a lifelong defense baseline called Anisotropic \& Isotropic Replay (AIR), which offers three advantages: (1) Isotropic replay ensures model consistency in the neighborhood distribution of new data, indirectly aligning the output preference between old and new tasks. (2) Anisotropic replay enables the model to learn a compromise data manifold with fresh mixed semantics for further replay constraints and potential future attacks. (3) A straightforward regularizer mitigates the 'plasticity-stability' trade-off by aligning model output between new and old tasks. Experiment results demonstrate that AIR can approximate or even exceed the empirical performance upper bounds achieved by Joint Training.



## **8. Security Allocation in Networked Control Systems under Stealthy Attacks**

eess.SY

12 pages, 3 figures, and 1 table, journal submission

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2308.16639v2) [paper-pdf](http://arxiv.org/pdf/2308.16639v2)

**Authors**: Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper considers the problem of security allocation in a networked control system under stealthy attacks. The system is comprised of interconnected subsystems represented by vertices. A malicious adversary selects a single vertex on which to conduct a stealthy data injection attack with the purpose of maximally disrupting a distant target vertex while remaining undetected. Defense resources against the adversary are allocated by a defender on several selected vertices. First, the objectives of the adversary and the defender with uncertain targets are formulated in a probabilistic manner, resulting in an expected worst-case impact of stealthy attacks. Next, we provide a graph-theoretic necessary and sufficient condition under which the cost for the defender and the expected worst-case impact of stealthy attacks are bounded. This condition enables the defender to restrict the admissible actions to dominating sets of the graph representing the network. Then, the security allocation problem is solved through a Stackelberg game-theoretic framework. Finally, the obtained results are validated through a numerical example of a 50-vertex networked control system.



## **9. ADVREPAIR:Provable Repair of Adversarial Attack**

cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01642v1) [paper-pdf](http://arxiv.org/pdf/2404.01642v1)

**Authors**: Zhiming Chi, Jianan Ma, Pengfei Yang, Cheng-Chao Huang, Renjue Li, Xiaowei Huang, Lijun Zhang

**Abstract**: Deep neural networks (DNNs) are increasingly deployed in safety-critical domains, but their vulnerability to adversarial attacks poses serious safety risks. Existing neuron-level methods using limited data lack efficacy in fixing adversaries due to the inherent complexity of adversarial attack mechanisms, while adversarial training, leveraging a large number of adversarial samples to enhance robustness, lacks provability. In this paper, we propose ADVREPAIR, a novel approach for provable repair of adversarial attacks using limited data. By utilizing formal verification, ADVREPAIR constructs patch modules that, when integrated with the original network, deliver provable and specialized repairs within the robustness neighborhood. Additionally, our approach incorporates a heuristic mechanism for assigning patch modules, allowing this defense against adversarial attacks to generalize to other inputs. ADVREPAIR demonstrates superior efficiency, scalability and repair success rate. Different from existing DNN repair methods, our repair can generalize to general inputs, thereby improving the robustness of the neural network globally, which indicates a significant breakthrough in the generalization capability of ADVREPAIR.



## **10. Multi-granular Adversarial Attacks against Black-box Neural Ranking Models**

cs.IR

Accepted by SIGIR 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01574v1) [paper-pdf](http://arxiv.org/pdf/2404.01574v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Adversarial ranking attacks have gained increasing attention due to their success in probing vulnerabilities, and, hence, enhancing the robustness, of neural ranking models. Conventional attack methods employ perturbations at a single granularity, e.g., word-level or sentence-level, to a target document. However, limiting perturbations to a single level of granularity may reduce the flexibility of creating adversarial examples, thereby diminishing the potential threat of the attack. Therefore, we focus on generating high-quality adversarial examples by incorporating multi-granular perturbations. Achieving this objective involves tackling a combinatorial explosion problem, which requires identifying an optimal combination of perturbations across all possible levels of granularity, positions, and textual pieces. To address this challenge, we transform the multi-granular adversarial attack into a sequential decision-making process, where perturbations in the next attack step are influenced by the perturbed document in the current attack step. Since the attack process can only access the final state without direct intermediate signals, we use reinforcement learning to perform multi-granular attacks. During the reinforcement learning process, two agents work cooperatively to identify multi-granular vulnerabilities as attack targets and organize perturbation candidates into a final perturbation sequence. Experimental results show that our attack method surpasses prevailing baselines in both attack effectiveness and imperceptibility.



## **11. MMCert: Provable Defense against Adversarial Attacks to Multi-modal Models**

cs.CV

To appear in CVPR'24

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.19080v3) [paper-pdf](http://arxiv.org/pdf/2403.19080v3)

**Authors**: Yanting Wang, Hongye Fu, Wei Zou, Jinyuan Jia

**Abstract**: Different from a unimodal model whose input is from a single modality, the input (called multi-modal input) of a multi-modal model is from multiple modalities such as image, 3D points, audio, text, etc. Similar to unimodal models, many existing studies show that a multi-modal model is also vulnerable to adversarial perturbation, where an attacker could add small perturbation to all modalities of a multi-modal input such that the multi-modal model makes incorrect predictions for it. Existing certified defenses are mostly designed for unimodal models, which achieve sub-optimal certified robustness guarantees when extended to multi-modal models as shown in our experimental results. In our work, we propose MMCert, the first certified defense against adversarial attacks to a multi-modal model. We derive a lower bound on the performance of our MMCert under arbitrary adversarial attacks with bounded perturbations to both modalities (e.g., in the context of auto-driving, we bound the number of changed pixels in both RGB image and depth image). We evaluate our MMCert using two benchmark datasets: one for the multi-modal road segmentation task and the other for the multi-modal emotion recognition task. Moreover, we compare our MMCert with a state-of-the-art certified defense extended from unimodal models. Our experimental results show that our MMCert outperforms the baseline.



## **12. Rumor Detection with a novel graph neural network approach**

cs.AI

10 pages, 5 figures

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16206v3) [paper-pdf](http://arxiv.org/pdf/2403.16206v3)

**Authors**: Tianrui Liu, Qi Cai, Changxin Xu, Bo Hong, Fanghao Ni, Yuxin Qiao, Tsungwei Yang

**Abstract**: The wide spread of rumors on social media has caused a negative impact on people's daily life, leading to potential panic, fear, and mental health problems for the public. How to debunk rumors as early as possible remains a challenging problem. Existing studies mainly leverage information propagation structure to detect rumors, while very few works focus on correlation among users that they may coordinate to spread rumors in order to gain large popularity. In this paper, we propose a new detection model, that jointly learns both the representations of user correlation and information propagation to detect rumors on social media. Specifically, we leverage graph neural networks to learn the representations of user correlation from a bipartite graph that describes the correlations between users and source tweets, and the representations of information propagation with a tree structure. Then we combine the learned representations from these two modules to classify the rumors. Since malicious users intend to subvert our model after deployment, we further develop a greedy attack scheme to analyze the cost of three adversarial attacks: graph attack, comment attack, and joint attack. Evaluation results on two public datasets illustrate that the proposed MODEL outperforms the state-of-the-art rumor detection models. We also demonstrate our method performs well for early rumor detection. Moreover, the proposed detection method is more robust to adversarial attacks compared to the best existing method. Importantly, we show that it requires a high cost for attackers to subvert user correlation pattern, demonstrating the importance of considering user correlation for rumor detection.



## **13. Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats**

cs.CR

Chen Wu and Xi Li are equal contribution. The corresponding author is  Jiaqi Wang

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2401.10375v2) [paper-pdf](http://arxiv.org/pdf/2401.10375v2)

**Authors**: Chen Wu, Xi Li, Jiaqi Wang

**Abstract**: Federated Learning (FL) addresses critical issues in machine learning related to data privacy and security, yet suffering from data insufficiency and imbalance under certain circumstances. The emergence of foundation models (FMs) offers potential solutions to the limitations of existing FL frameworks, e.g., by generating synthetic data for model initialization. However, due to the inherent safety concerns of FMs, integrating FMs into FL could introduce new risks, which remains largely unexplored. To address this gap, we conduct the first investigation on the vulnerability of FM integrated FL (FM-FL) under adversarial threats. Based on a unified framework of FM-FL, we introduce a novel attack strategy that exploits safety issues of FM to compromise FL client models. Through extensive experiments with well-known models and benchmark datasets in both image and text domains, we reveal the high susceptibility of the FM-FL to this new threat under various FL configurations. Furthermore, we find that existing FL defense strategies offer limited protection against this novel attack approach. This research highlights the critical need for enhanced security measures in FL in the era of FMs.



## **14. Can Biases in ImageNet Models Explain Generalization?**

cs.CV

Accepted at CVPR2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01509v1) [paper-pdf](http://arxiv.org/pdf/2404.01509v1)

**Authors**: Paul Gavrikov, Janis Keuper

**Abstract**: The robust generalization of models to rare, in-distribution (ID) samples drawn from the long tail of the training distribution and to out-of-training-distribution (OOD) samples is one of the major challenges of current deep learning methods. For image classification, this manifests in the existence of adversarial attacks, the performance drops on distorted images, and a lack of generalization to concepts such as sketches. The current understanding of generalization in neural networks is very limited, but some biases that differentiate models from human vision have been identified and might be causing these limitations. Consequently, several attempts with varying success have been made to reduce these biases during training to improve generalization. We take a step back and sanity-check these attempts. Fixing the architecture to the well-established ResNet-50, we perform a large-scale study on 48 ImageNet models obtained via different training methods to understand how and if these biases - including shape bias, spectral biases, and critical bands - interact with generalization. Our extensive study results reveal that contrary to previous findings, these biases are insufficient to accurately predict the generalization of a model holistically. We provide access to all checkpoints and evaluation code at https://github.com/paulgavrikov/biases_vs_generalization



## **15. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2403.05030v3) [paper-pdf](http://arxiv.org/pdf/2403.05030v3)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use it to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.



## **16. Robust One-Class Classification with Signed Distance Function using 1-Lipschitz Neural Networks**

cs.LG

27 pages, 11 figures, International Conference on Machine Learning  2023, (ICML 2023)

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2303.01978v2) [paper-pdf](http://arxiv.org/pdf/2303.01978v2)

**Authors**: Louis Bethune, Paul Novello, Thibaut Boissin, Guillaume Coiffier, Mathieu Serrurier, Quentin Vincenot, Andres Troya-Galvis

**Abstract**: We propose a new method, dubbed One Class Signed Distance Function (OCSDF), to perform One Class Classification (OCC) by provably learning the Signed Distance Function (SDF) to the boundary of the support of any distribution. The distance to the support can be interpreted as a normality score, and its approximation using 1-Lipschitz neural networks provides robustness bounds against $l2$ adversarial attacks, an under-explored weakness of deep learning-based OCC algorithms. As a result, OCSDF comes with a new metric, certified AUROC, that can be computed at the same cost as any classical AUROC. We show that OCSDF is competitive against concurrent methods on tabular and image data while being way more robust to adversarial attacks, illustrating its theoretical properties. Finally, as exploratory research perspectives, we theoretically and empirically show how OCSDF connects OCC with image generation and implicit neural surface parametrization. Our code is available at https://github.com/Algue-Rythme/OneClassMetricLearning



## **17. The twin peaks of learning neural networks**

cs.LG

37 pages, 31 figures

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2401.12610v2) [paper-pdf](http://arxiv.org/pdf/2401.12610v2)

**Authors**: Elizaveta Demyanenko, Christoph Feinauer, Enrico M. Malatesta, Luca Saglietti

**Abstract**: Recent works demonstrated the existence of a double-descent phenomenon for the generalization error of neural networks, where highly overparameterized models escape overfitting and achieve good test performance, at odds with the standard bias-variance trade-off described by statistical learning theory. In the present work, we explore a link between this phenomenon and the increase of complexity and sensitivity of the function represented by neural networks. In particular, we study the Boolean mean dimension (BMD), a metric developed in the context of Boolean function analysis. Focusing on a simple teacher-student setting for the random feature model, we derive a theoretical analysis based on the replica method that yields an interpretable expression for the BMD, in the high dimensional regime where the number of data points, the number of features, and the input size grow to infinity. We find that, as the degree of overparameterization of the network is increased, the BMD reaches an evident peak at the interpolation threshold, in correspondence with the generalization error peak, and then slowly approaches a low asymptotic value. The same phenomenology is then traced in numerical experiments with different model classes and training setups. Moreover, we find empirically that adversarially initialized models tend to show higher BMD values, and that models that are more robust to adversarial attacks exhibit a lower BMD.



## **18. Foundations of Cyber Resilience: The Confluence of Game, Control, and Learning Theories**

eess.SY

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01205v1) [paper-pdf](http://arxiv.org/pdf/2404.01205v1)

**Authors**: Quanyan Zhu

**Abstract**: Cyber resilience is a complementary concept to cybersecurity, focusing on the preparation, response, and recovery from cyber threats that are challenging to prevent. Organizations increasingly face such threats in an evolving cyber threat landscape. Understanding and establishing foundations for cyber resilience provide a quantitative and systematic approach to cyber risk assessment, mitigation policy evaluation, and risk-informed defense design. A systems-scientific view toward cyber risks provides holistic and system-level solutions. This chapter starts with a systemic view toward cyber risks and presents the confluence of game theory, control theory, and learning theories, which are three major pillars for the design of cyber resilience mechanisms to counteract increasingly sophisticated and evolving threats in our networks and organizations. Game and control theoretic methods provide a set of modeling frameworks to capture the strategic and dynamic interactions between defenders and attackers. Control and learning frameworks together provide a feedback-driven mechanism that enables autonomous and adaptive responses to threats. Game and learning frameworks offer a data-driven approach to proactively reason about adversarial behaviors and resilient strategies. The confluence of the three lays the theoretical foundations for the analysis and design of cyber resilience. This chapter presents various theoretical paradigms, including dynamic asymmetric games, moving horizon control, conjectural learning, and meta-learning, as recent advances at the intersection. This chapter concludes with future directions and discussions of the role of neurosymbolic learning and the synergy between foundation models and game models in cyber resilience.



## **19. The Best Defense is Attack: Repairing Semantics in Textual Adversarial Examples**

cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2305.04067v2) [paper-pdf](http://arxiv.org/pdf/2305.04067v2)

**Authors**: Heng Yang, Ke Li

**Abstract**: Recent studies have revealed the vulnerability of pre-trained language models to adversarial attacks. Existing adversarial defense techniques attempt to reconstruct adversarial examples within feature or text spaces. However, these methods struggle to effectively repair the semantics in adversarial examples, resulting in unsatisfactory performance and limiting their practical utility. To repair the semantics in adversarial examples, we introduce a novel approach named Reactive Perturbation Defocusing (Rapid). Rapid employs an adversarial detector to identify fake labels of adversarial examples and leverage adversarial attackers to repair the semantics in adversarial examples. Our extensive experimental results conducted on four public datasets, convincingly demonstrate the effectiveness of Rapid in various adversarial attack scenarios. To address the problem of defense performance validation in previous works, we provide a demonstration of adversarial detection and repair based on our work, which can be easily evaluated at https://tinyurl.com/22ercuf8.



## **20. Poisoning Decentralized Collaborative Recommender System and Its Countermeasures**

cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01177v1) [paper-pdf](http://arxiv.org/pdf/2404.01177v1)

**Authors**: Ruiqi Zheng, Liang Qu, Tong Chen, Kai Zheng, Yuhui Shi, Hongzhi Yin

**Abstract**: To make room for privacy and efficiency, the deployment of many recommender systems is experiencing a shift from central servers to personal devices, where the federated recommender systems (FedRecs) and decentralized collaborative recommender systems (DecRecs) are arguably the two most representative paradigms. While both leverage knowledge (e.g., gradients) sharing to facilitate learning local models, FedRecs rely on a central server to coordinate the optimization process, yet in DecRecs, the knowledge sharing directly happens between clients. Knowledge sharing also opens a backdoor for model poisoning attacks, where adversaries disguise themselves as benign clients and disseminate polluted knowledge to achieve malicious goals like promoting an item's exposure rate. Although research on such poisoning attacks provides valuable insights into finding security loopholes and corresponding countermeasures, existing attacks mostly focus on FedRecs, and are either inapplicable or ineffective for DecRecs. Compared with FedRecs where the tampered information can be universally distributed to all clients once uploaded to the cloud, each adversary in DecRecs can only communicate with neighbor clients of a small size, confining its impact to a limited range. To fill the gap, we present a novel attack method named Poisoning with Adaptive Malicious Neighbors (PAMN). With item promotion in top-K recommendation as the attack objective, PAMN effectively boosts target items' ranks with several adversaries that emulate benign clients and transfers adaptively crafted gradients conditioned on each adversary's neighbors. Moreover, with the vulnerabilities of DecRecs uncovered, a dedicated defensive mechanism based on user-level gradient clipping with sparsified updating is proposed. Extensive experiments demonstrate the effectiveness of the poisoning attack and the robustness of our defensive mechanism.



## **21. The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness**

cs.LG

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01356v1) [paper-pdf](http://arxiv.org/pdf/2404.01356v1)

**Authors**: Xuran Li, Peng Wu, Yanting Chen, Xingjun Ma, Zhen Zhang, Kaixiang Dong

**Abstract**: Deep neural networks (DNNs) are known to be sensitive to adversarial input perturbations, leading to a reduction in either prediction accuracy or individual fairness. To jointly characterize the susceptibility of prediction accuracy and individual fairness to adversarial perturbations, we introduce a novel robustness definition termed robust accurate fairness. Informally, robust accurate fairness requires that predictions for an instance and its similar counterparts consistently align with the ground truth when subjected to input perturbations. We propose an adversarial attack approach dubbed RAFair to expose false or biased adversarial defects in DNN, which either deceive accuracy or compromise individual fairness. Then, we show that such adversarial instances can be effectively addressed by carefully designed benign perturbations, correcting their predictions to be accurate and fair. Our work explores the double-edged sword of input perturbations to robust accurate fairness in DNN and the potential of using benign perturbations to correct adversarial instances.



## **22. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2310.00322v3) [paper-pdf](http://arxiv.org/pdf/2310.00322v3)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.



## **23. LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer**

cs.CV

14 pages, 3 figures. Accepted to AAAI 2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2312.09935v2) [paper-pdf](http://arxiv.org/pdf/2312.09935v2)

**Authors**: Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu

**Abstract**: Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.



## **24. StyleFool: Fooling Video Classification Systems via Style Transfer**

cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2203.16000v4) [paper-pdf](http://arxiv.org/pdf/2203.16000v4)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.



## **25. BadPart: Unified Black-box Adversarial Patch Attacks against Pixel-wise Regression Tasks**

cs.CV

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.00924v1) [paper-pdf](http://arxiv.org/pdf/2404.00924v1)

**Authors**: Zhiyuan Cheng, Zhaoyi Liu, Tengda Guo, Shiwei Feng, Dongfang Liu, Mingjie Tang, Xiangyu Zhang

**Abstract**: Pixel-wise regression tasks (e.g., monocular depth estimation (MDE) and optical flow estimation (OFE)) have been widely involved in our daily life in applications like autonomous driving, augmented reality and video composition. Although certain applications are security-critical or bear societal significance, the adversarial robustness of such models are not sufficiently studied, especially in the black-box scenario. In this work, we introduce the first unified black-box adversarial patch attack framework against pixel-wise regression tasks, aiming to identify the vulnerabilities of these models under query-based black-box attacks. We propose a novel square-based adversarial patch optimization framework and employ probabilistic square sampling and score-based gradient estimation techniques to generate the patch effectively and efficiently, overcoming the scalability problem of previous black-box patch attacks. Our attack prototype, named BadPart, is evaluated on both MDE and OFE tasks, utilizing a total of 7 models. BadPart surpasses 3 baseline methods in terms of both attack performance and efficiency. We also apply BadPart on the Google online service for portrait depth estimation, causing 43.5% relative distance error with 50K queries. State-of-the-art (SOTA) countermeasures cannot defend our attack effectively.



## **26. An Embarrassingly Simple Defense Against Backdoor Attacks On SSL**

cs.CV

10 pages, 5 figures

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2403.15918v2) [paper-pdf](http://arxiv.org/pdf/2403.15918v2)

**Authors**: Aryan Satpathy, Nilaksh Nilaksh, Dhruva Rajwade

**Abstract**: Self Supervised Learning (SSL) has emerged as a powerful paradigm to tackle data landscapes with absence of human supervision. The ability to learn meaningful tasks without the use of labeled data makes SSL a popular method to manage large chunks of data in the absence of labels. However, recent work indicates SSL to be vulnerable to backdoor attacks, wherein models can be controlled, possibly maliciously, to suit an adversary's motives. Li et. al (2022) introduce a novel frequency-based backdoor attack: CTRL. They show that CTRL can be used to efficiently and stealthily gain control over a victim's model trained using SSL. In this work, we devise two defense strategies against frequency-based attacks in SSL: One applicable before model training and the second to be applied during model inference. Our first contribution utilizes the invariance property of the downstream task to defend against backdoor attacks in a generalizable fashion. We observe the ASR (Attack Success Rate) to reduce by over 60% across experiments. Our Inference-time defense relies on evasiveness of the attack and uses the luminance channel to defend against attacks. Using object classification as the downstream task for SSL, we demonstrate successful defense strategies that do not require re-training of the model. Code is available at https://github.com/Aryan-Satpathy/Backdoor.



## **27. Machine Learning Robustness: A Primer**

cs.LG

arXiv admin note: text overlap with arXiv:2305.10862 by other authors

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.00897v1) [paper-pdf](http://arxiv.org/pdf/2404.00897v1)

**Authors**: Houssem Ben Braiek, Foutse Khomh

**Abstract**: This chapter explores the foundational concept of robustness in Machine Learning (ML) and its integral role in establishing trustworthiness in Artificial Intelligence (AI) systems. The discussion begins with a detailed definition of robustness, portraying it as the ability of ML models to maintain stable performance across varied and unexpected environmental conditions. ML robustness is dissected through several lenses: its complementarity with generalizability; its status as a requirement for trustworthy AI; its adversarial vs non-adversarial aspects; its quantitative metrics; and its indicators such as reproducibility and explainability. The chapter delves into the factors that impede robustness, such as data bias, model complexity, and the pitfalls of underspecified ML pipelines. It surveys key techniques for robustness assessment from a broad perspective, including adversarial attacks, encompassing both digital and physical realms. It covers non-adversarial data shifts and nuances of Deep Learning (DL) software testing methodologies. The discussion progresses to explore amelioration strategies for bolstering robustness, starting with data-centric approaches like debiasing and augmentation. Further examination includes a variety of model-centric methods such as transfer learning, adversarial training, and randomized smoothing. Lastly, post-training methods are discussed, including ensemble techniques, pruning, and model repairs, emerging as cost-effective strategies to make models more resilient against the unpredictable. This chapter underscores the ongoing challenges and limitations in estimating and achieving ML robustness by existing approaches. It offers insights and directions for future research on this crucial concept, as a prerequisite for trustworthy AI systems.



## **28. Privacy Re-identification Attacks on Tabular GANs**

cs.CR

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.00696v1) [paper-pdf](http://arxiv.org/pdf/2404.00696v1)

**Authors**: Abdallah Alshantti, Adil Rasheed, Frank Westad

**Abstract**: Generative models are subject to overfitting and thus may potentially leak sensitive information from the training data. In this work. we investigate the privacy risks that can potentially arise from the use of generative adversarial networks (GANs) for creating tabular synthetic datasets. For the purpose, we analyse the effects of re-identification attacks on synthetic data, i.e., attacks which aim at selecting samples that are predicted to correspond to memorised training samples based on their proximity to the nearest synthetic records. We thus consider multiple settings where different attackers might have different access levels or knowledge of the generative model and predictive, and assess which information is potentially most useful for launching more successful re-identification attacks. In doing so we also consider the situation for which re-identification attacks are formulated as reconstruction attacks, i.e., the situation where an attacker uses evolutionary multi-objective optimisation for perturbing synthetic samples closer to the training space. The results indicate that attackers can indeed pose major privacy risks by selecting synthetic samples that are likely representative of memorised training samples. In addition, we notice that privacy threats considerably increase when the attacker either has knowledge or has black-box access to the generative models. We also find that reconstruction attacks through multi-objective optimisation even increase the risk of identifying confidential samples.



## **29. Model-less Is the Best Model: Generating Pure Code Implementations to Replace On-Device DL Models**

cs.SE

Accepted by the ACM SIGSOFT International Symposium on Software  Testing and Analysis (ISSTA2024)

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2403.16479v2) [paper-pdf](http://arxiv.org/pdf/2403.16479v2)

**Authors**: Mingyi Zhou, Xiang Gao, Pei Liu, John Grundy, Chunyang Chen, Xiao Chen, Li Li

**Abstract**: Recent studies show that deployed deep learning (DL) models such as those of Tensor Flow Lite (TFLite) can be easily extracted from real-world applications and devices by attackers to generate many kinds of attacks like adversarial attacks. Although securing deployed on-device DL models has gained increasing attention, no existing methods can fully prevent the aforementioned threats. Traditional software protection techniques have been widely explored, if on-device models can be implemented using pure code, such as C++, it will open the possibility of reusing existing software protection techniques. However, due to the complexity of DL models, there is no automatic method that can translate the DL models to pure code. To fill this gap, we propose a novel method, CustomDLCoder, to automatically extract the on-device model information and synthesize a customized executable program for a wide range of DL models. CustomDLCoder first parses the DL model, extracts its backend computing units, configures the computing units to a graph, and then generates customized code to implement and deploy the ML solution without explicit model representation. The synthesized program hides model information for DL deployment environments since it does not need to retain explicit model representation, preventing many attacks on the DL model. In addition, it improves ML performance because the customized code removes model parsing and preprocessing steps and only retains the data computing process. Our experimental results show that CustomDLCoder improves model security by disabling on-device model sniffing. Compared with the original on-device platform (i.e., TFLite), our method can accelerate model inference by 21.8% and 24.3% on x86-64 and ARM64 platforms, respectively. Most importantly, it can significantly reduce memory consumption by 68.8% and 36.0% on x86-64 and ARM64 platforms, respectively.



## **30. Embodied Active Defense: Leveraging Recurrent Feedback to Counter Adversarial Patches**

cs.CV

27pages

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.00540v1) [paper-pdf](http://arxiv.org/pdf/2404.00540v1)

**Authors**: Lingxuan Wu, Xiao Yang, Yinpeng Dong, Liuwei Xie, Hang Su, Jun Zhu

**Abstract**: The vulnerability of deep neural networks to adversarial patches has motivated numerous defense strategies for boosting model robustness. However, the prevailing defenses depend on single observation or pre-established adversary information to counter adversarial patches, often failing to be confronted with unseen or adaptive adversarial attacks and easily exhibiting unsatisfying performance in dynamic 3D environments. Inspired by active human perception and recurrent feedback mechanisms, we develop Embodied Active Defense (EAD), a proactive defensive strategy that actively contextualizes environmental information to address misaligned adversarial patches in 3D real-world settings. To achieve this, EAD develops two central recurrent sub-modules, i.e., a perception module and a policy module, to implement two critical functions of active vision. These models recurrently process a series of beliefs and observations, facilitating progressive refinement of their comprehension of the target object and enabling the development of strategic actions to counter adversarial patches in 3D environments. To optimize learning efficiency, we incorporate a differentiable approximation of environmental dynamics and deploy patches that are agnostic to the adversary strategies. Extensive experiments demonstrate that EAD substantially enhances robustness against a variety of patches within just a few steps through its action policy in safety-critical tasks (e.g., face recognition and object detection), without compromising standard accuracy. Furthermore, due to the attack-agnostic characteristic, EAD facilitates excellent generalization to unseen attacks, diminishing the averaged attack success rate by 95 percent across a range of unseen adversarial attacks.



## **31. AttackNet: Enhancing Biometric Security via Tailored Convolutional Neural Network Architectures for Liveness Detection**

cs.CV

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2402.03769v2) [paper-pdf](http://arxiv.org/pdf/2402.03769v2)

**Authors**: Oleksandr Kuznetsov, Dmytro Zakharov, Emanuele Frontoni, Andrea Maranesi

**Abstract**: Biometric security is the cornerstone of modern identity verification and authentication systems, where the integrity and reliability of biometric samples is of paramount importance. This paper introduces AttackNet, a bespoke Convolutional Neural Network architecture, meticulously designed to combat spoofing threats in biometric systems. Rooted in deep learning methodologies, this model offers a layered defense mechanism, seamlessly transitioning from low-level feature extraction to high-level pattern discernment. Three distinctive architectural phases form the crux of the model, each underpinned by judiciously chosen activation functions, normalization techniques, and dropout layers to ensure robustness and resilience against adversarial attacks. Benchmarking our model across diverse datasets affirms its prowess, showcasing superior performance metrics in comparison to contemporary models. Furthermore, a detailed comparative analysis accentuates the model's efficacy, drawing parallels with prevailing state-of-the-art methodologies. Through iterative refinement and an informed architectural strategy, AttackNet underscores the potential of deep learning in safeguarding the future of biometric security.



## **32. Bidirectional Consistency Models**

cs.LG

40 pages, 25 figures

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2403.18035v2) [paper-pdf](http://arxiv.org/pdf/2403.18035v2)

**Authors**: Liangchen Li, Jiajun He

**Abstract**: Diffusion models (DMs) are capable of generating remarkably high-quality samples by iteratively denoising a random vector, a process that corresponds to moving along the probability flow ordinary differential equation (PF ODE). Interestingly, DMs can also invert an input image to noise by moving backward along the PF ODE, a key operation for downstream tasks such as interpolation and image editing. However, the iterative nature of this process restricts its speed, hindering its broader application. Recently, Consistency Models (CMs) have emerged to address this challenge by approximating the integral of the PF ODE, largely reducing the number of iterations. Yet, the absence of an explicit ODE solver complicates the inversion process. To resolve this, we introduce the Bidirectional Consistency Model (BCM), which learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. Notably, our proposed method enables one-step generation and inversion while also allowing the use of additional steps to enhance generation quality or reduce reconstruction error. Furthermore, by leveraging our model's bidirectional consistency, we introduce a sampling strategy that can enhance FID while preserving the generated image content. We further showcase our model's capabilities in several downstream tasks, such as interpolation and inpainting, and present demonstrations of potential applications, including blind restoration of compressed images and defending black-box adversarial attacks.



## **33. STBA: Towards Evaluating the Robustness of DNNs for Query-Limited Black-box Scenario**

cs.CV

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2404.00362v1) [paper-pdf](http://arxiv.org/pdf/2404.00362v1)

**Authors**: Renyang Liu, Kwok-Yan Lam, Wei Zhou, Sixing Wu, Jun Zhao, Dongting Hu, Mingming Gong

**Abstract**: Many attack techniques have been proposed to explore the vulnerability of DNNs and further help to improve their robustness. Despite the significant progress made recently, existing black-box attack methods still suffer from unsatisfactory performance due to the vast number of queries needed to optimize desired perturbations. Besides, the other critical challenge is that adversarial examples built in a noise-adding manner are abnormal and struggle to successfully attack robust models, whose robustness is enhanced by adversarial training against small perturbations. There is no doubt that these two issues mentioned above will significantly increase the risk of exposure and result in a failure to dig deeply into the vulnerability of DNNs. Hence, it is necessary to evaluate DNNs' fragility sufficiently under query-limited settings in a non-additional way. In this paper, we propose the Spatial Transform Black-box Attack (STBA), a novel framework to craft formidable adversarial examples in the query-limited scenario. Specifically, STBA introduces a flow field to the high-frequency part of clean images to generate adversarial examples and adopts the following two processes to enhance their naturalness and significantly improve the query efficiency: a) we apply an estimated flow field to the high-frequency part of clean images to generate adversarial examples instead of introducing external noise to the benign image, and b) we leverage an efficient gradient estimation method based on a batch of samples to optimize such an ideal flow field under query-limited settings. Compared to existing score-based black-box baselines, extensive experiments indicated that STBA could effectively improve the imperceptibility of the adversarial examples and remarkably boost the attack success rate under query-limited settings.



## **34. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

cs.CL

Code/data: https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2402.17916v2) [paper-pdf](http://arxiv.org/pdf/2402.17916v2)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure, offering a nuanced view into model's limitation.



## **35. On Inherent Adversarial Robustness of Active Vision Systems**

cs.CV

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2404.00185v1) [paper-pdf](http://arxiv.org/pdf/2404.00185v1)

**Authors**: Amitangshu Mukherjee, Timur Ibrayev, Kaushik Roy

**Abstract**: Current Deep Neural Networks are vulnerable to adversarial examples, which alter their predictions by adding carefully crafted noise. Since human eyes are robust to such inputs, it is possible that the vulnerability stems from the standard way of processing inputs in one shot by processing every pixel with the same importance. In contrast, neuroscience suggests that the human vision system can differentiate salient features by (1) switching between multiple fixation points (saccades) and (2) processing the surrounding with a non-uniform external resolution (foveation). In this work, we advocate that the integration of such active vision mechanisms into current deep learning systems can offer robustness benefits. Specifically, we empirically demonstrate the inherent robustness of two active vision methods - GFNet and FALcon - under a black box threat model. By learning and inferencing based on downsampled glimpses obtained from multiple distinct fixation points within an input, we show that these active methods achieve (2-3) times greater robustness compared to a standard passive convolutional network under state-of-the-art adversarial attacks. More importantly, we provide illustrative and interpretable visualization analysis that demonstrates how performing inference from distinct fixation points makes active vision methods less vulnerable to malicious inputs.



## **36. Deepfake Sentry: Harnessing Ensemble Intelligence for Resilient Detection and Generalisation**

cs.CV

16 pages, 1 figure, U.P.B. Sci. Bull., Series C, Vol. 85, Iss. 4,  2023

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2404.00114v1) [paper-pdf](http://arxiv.org/pdf/2404.00114v1)

**Authors**: Liviu-Daniel Ştefan, Dan-Cristian Stanciu, Mihai Dogariu, Mihai Gabriel Constantin, Andrei Cosmin Jitaru, Bogdan Ionescu

**Abstract**: Recent advancements in Generative Adversarial Networks (GANs) have enabled photorealistic image generation with high quality. However, the malicious use of such generated media has raised concerns regarding visual misinformation. Although deepfake detection research has demonstrated high accuracy, it is vulnerable to advances in generation techniques and adversarial iterations on detection countermeasures. To address this, we propose a proactive and sustainable deepfake training augmentation solution that introduces artificial fingerprints into models. We achieve this by employing an ensemble learning approach that incorporates a pool of autoencoders that mimic the effect of the artefacts introduced by the deepfake generator models. Experiments on three datasets reveal that our proposed ensemble autoencoder-based data augmentation learning approach offers improvements in terms of generalisation, resistance against basic data perturbations such as noise, blurring, sharpness enhancement, and affine transforms, resilience to commonly used lossy compression algorithms such as JPEG, and enhanced resistance against adversarial attacks.



## **37. LipSim: A Provably Robust Perceptual Similarity Metric**

cs.CV

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2310.18274v2) [paper-pdf](http://arxiv.org/pdf/2310.18274v2)

**Authors**: Sara Ghazanfari, Alexandre Araujo, Prashanth Krishnamurthy, Farshad Khorrami, Siddharth Garg

**Abstract**: Recent years have seen growing interest in developing and applying perceptual similarity metrics. Research has shown the superiority of perceptual metrics over pixel-wise metrics in aligning with human perception and serving as a proxy for the human visual system. On the other hand, as perceptual metrics rely on neural networks, there is a growing concern regarding their resilience, given the established vulnerability of neural networks to adversarial attacks. It is indeed logical to infer that perceptual metrics may inherit both the strengths and shortcomings of neural networks. In this work, we demonstrate the vulnerability of state-of-the-art perceptual similarity metrics based on an ensemble of ViT-based feature extractors to adversarial attacks. We then propose a framework to train a robust perceptual similarity metric called LipSim (Lipschitz Similarity Metric) with provable guarantees. By leveraging 1-Lipschitz neural networks as the backbone, LipSim provides guarded areas around each data point and certificates for all perturbations within an $\ell_2$ ball. Finally, a comprehensive set of experiments shows the performance of LipSim in terms of natural and certified scores and on the image retrieval application. The code is available at https://github.com/SaraGhazanfari/LipSim.



## **38. Selective Attention-based Modulation for Continual Learning**

cs.CV

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2403.20086v1) [paper-pdf](http://arxiv.org/pdf/2403.20086v1)

**Authors**: Giovanni Bellitto, Federica Proietto Salanitri, Matteo Pennisi, Matteo Boschini, Angelo Porrello, Simone Calderara, Simone Palazzo, Concetto Spampinato

**Abstract**: We present SAM, a biologically-plausible selective attention-driven modulation approach to enhance classification models in a continual learning setting. Inspired by neurophysiological evidence that the primary visual cortex does not contribute to object manifold untangling for categorization and that primordial attention biases are still embedded in the modern brain, we propose to employ auxiliary saliency prediction features as a modulation signal to drive and stabilize the learning of a sequence of non-i.i.d. classification tasks. Experimental results confirm that SAM effectively enhances the performance (in some cases up to about twenty percent points) of state-of-the-art continual learning methods, both in class-incremental and task-incremental settings. Moreover, we show that attention-based modulation successfully encourages the learning of features that are more robust to the presence of spurious features and to adversarial attacks than baseline methods. Code is available at: https://github.com/perceivelab/SAM.



## **39. Strong Transferable Adversarial Attacks via Ensembled Asymptotically Normal Distribution Learning**

cs.LG

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2209.11964v2) [paper-pdf](http://arxiv.org/pdf/2209.11964v2)

**Authors**: Zhengwei Fang, Rui Wang, Tao Huang, Liping Jing

**Abstract**: Strong adversarial examples are crucial for evaluating and enhancing the robustness of deep neural networks. However, the performance of popular attacks is usually sensitive, for instance, to minor image transformations, stemming from limited information -- typically only one input example, a handful of white-box source models, and undefined defense strategies. Hence, the crafted adversarial examples are prone to overfit the source model, which hampers their transferability to unknown architectures. In this paper, we propose an approach named Multiple Asymptotically Normal Distribution Attacks (MultiANDA) which explicitly characterize adversarial perturbations from a learned distribution. Specifically, we approximate the posterior distribution over the perturbations by taking advantage of the asymptotic normality property of stochastic gradient ascent (SGA), then employ the deep ensemble strategy as an effective proxy for Bayesian marginalization in this process, aiming to estimate a mixture of Gaussians that facilitates a more thorough exploration of the potential optimization space. The approximated posterior essentially describes the stationary distribution of SGA iterations, which captures the geometric information around the local optimum. Thus, MultiANDA allows drawing an unlimited number of adversarial perturbations for each input and reliably maintains the transferability. Our proposed method outperforms ten state-of-the-art black-box attacks on deep learning models with or without defenses through extensive experiments on seven normally trained and seven defense models.



## **40. An Anomaly Behavior Analysis Framework for Securing Autonomous Vehicle Perception**

cs.RO

20th ACS/IEEE International Conference on Computer Systems and  Applications (Accepted for publication)

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2310.05041v2) [paper-pdf](http://arxiv.org/pdf/2310.05041v2)

**Authors**: Murad Mehrab Abrar, Salim Hariri

**Abstract**: As a rapidly growing cyber-physical platform, Autonomous Vehicles (AVs) are encountering more security challenges as their capabilities continue to expand. In recent years, adversaries are actively targeting the perception sensors of autonomous vehicles with sophisticated attacks that are not easily detected by the vehicles' control systems. This work proposes an Anomaly Behavior Analysis approach to detect a perception sensor attack against an autonomous vehicle. The framework relies on temporal features extracted from a physics-based autonomous vehicle behavior model to capture the normal behavior of vehicular perception in autonomous driving. By employing a combination of model-based techniques and machine learning algorithms, the proposed framework distinguishes between normal and abnormal vehicular perception behavior. To demonstrate the application of the framework in practice, we performed a depth camera attack experiment on an autonomous vehicle testbed and generated an extensive dataset. We validated the effectiveness of the proposed framework using this real-world data and released the dataset for public access. To our knowledge, this dataset is the first of its kind and will serve as a valuable resource for the research community in evaluating their intrusion detection techniques effectively.



## **41. Evolving Assembly Code in an Adversarial Environment**

cs.NE

9 pages, 5 figures, 6 listings

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19489v1) [paper-pdf](http://arxiv.org/pdf/2403.19489v1)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve assembly code for the CodeGuru competition. The competition's goal is to create a survivor -- an assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. In addition, we compare our approach with a Large-Language Model, demonstrating that the latter cannot generate a survivor that can win at any competition. This work has important applications for cyber-security, as we utilize evolution to detect weaknesses in survivors. The assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.



## **42. Cloudy with a Chance of Cyberattacks: Dangling Resources Abuse on Cloud Platforms**

cs.NI

17 pages, 29 figures, to be published in NSDI'24: Proceedings of the  21st USENIX Symposium on Networked Systems Design and Implementation

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19368v1) [paper-pdf](http://arxiv.org/pdf/2403.19368v1)

**Authors**: Jens Frieß, Tobias Gattermayer, Nethanel Gelernter, Haya Schulmann, Michael Waidner

**Abstract**: Recent works showed that it is feasible to hijack resources on cloud platforms. In such hijacks, attackers can take over released resources that belong to legitimate organizations. It was proposed that adversaries could abuse these resources to carry out attacks against customers of the hijacked services, e.g., through malware distribution. However, to date, no research has confirmed the existence of these attacks. We identify, for the first time, real-life hijacks of cloud resources. This yields a number of surprising and important insights. First, contrary to previous assumption that attackers primarily target IP addresses, our findings reveal that the type of resource is not the main consideration in a hijack. Attackers focus on hijacking records that allow them to determine the resource by entering freetext. The costs and overhead of hijacking such records are much lower than those of hijacking IP addresses, which are randomly selected from a large pool. Second, identifying hijacks poses a substantial challenge. Monitoring resource changes, e.g., changes in content, is insufficient, since such changes could also be legitimate. Retrospective analysis of digital assets to identify hijacks is also arduous due to the immense volume of data involved and the absence of indicators to search for. To address this challenge, we develop a novel approach that involves analyzing data from diverse sources to effectively differentiate between malicious and legitimate modifications. Our analysis has revealed 20,904 instances of hijacked resources on popular cloud platforms. While some hijacks are short-lived (up to 15 days), 1/3 persist for more than 65 days. We study how attackers abuse the hijacked resources and find that, in contrast to the threats considered in previous work, the majority of the abuse (75%) is blackhat search engine optimization.



## **43. Data-free Defense of Black Box Models Against Adversarial Attacks**

cs.LG

CVPR Workshop (Under Review)

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2211.01579v3) [paper-pdf](http://arxiv.org/pdf/2211.01579v3)

**Authors**: Gaurav Kumar Nayak, Inder Khatri, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Several companies often safeguard their trained deep models (i.e., details of architecture, learnt weights, training details etc.) from third-party users by exposing them only as black boxes through APIs. Moreover, they may not even provide access to the training data due to proprietary reasons or sensitivity concerns. In this work, we propose a novel defense mechanism for black box models against adversarial attacks in a data-free set up. We construct synthetic data via generative model and train surrogate network using model stealing techniques. To minimize adversarial contamination on perturbed samples, we propose 'wavelet noise remover' (WNR) that performs discrete wavelet decomposition on input images and carefully select only a few important coefficients determined by our 'wavelet coefficient selection module' (WCSM). To recover the high-frequency content of the image after noise removal via WNR, we further train a 'regenerator' network with an objective to retrieve the coefficients such that the reconstructed image yields similar to original predictions on the surrogate model. At test time, WNR combined with trained regenerator network is prepended to the black box network, resulting in a high boost in adversarial accuracy. Our method improves the adversarial accuracy on CIFAR-10 by 38.98% and 32.01% on state-of-the-art Auto Attack compared to baseline, even when the attacker uses surrogate architecture (Alexnet-half and Alexnet) similar to the black box architecture (Alexnet) with same model stealing strategy as defender. The code is available at https://github.com/vcl-iisc/data-free-black-box-defense



## **44. Feature Unlearning for Pre-trained GANs and VAEs**

cs.CV

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2303.05699v4) [paper-pdf](http://arxiv.org/pdf/2303.05699v4)

**Authors**: Saemi Moon, Seunghyuk Cho, Dongwoo Kim

**Abstract**: We tackle the problem of feature unlearning from a pre-trained image generative model: GANs and VAEs. Unlike a common unlearning task where an unlearning target is a subset of the training set, we aim to unlearn a specific feature, such as hairstyle from facial images, from the pre-trained generative models. As the target feature is only presented in a local region of an image, unlearning the entire image from the pre-trained model may result in losing other details in the remaining region of the image. To specify which features to unlearn, we collect randomly generated images that contain the target features. We then identify a latent representation corresponding to the target feature and then use the representation to fine-tune the pre-trained model. Through experiments on MNIST, CelebA, and FFHQ datasets, we show that target features are successfully removed while keeping the fidelity of the original models. Further experiments with an adversarial attack show that the unlearned model is more robust under the presence of malicious parties.



## **45. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2404.01318v1) [paper-pdf](http://arxiv.org/pdf/2404.01318v1)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) a new jailbreaking dataset containing 100 unique behaviors, which we call JBB-Behaviors; (2) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.



## **46. Data Poisoning for In-context Learning**

cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2402.02160v2) [paper-pdf](http://arxiv.org/pdf/2402.02160v2)

**Authors**: Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang

**Abstract**: In the domain of large language models (LLMs), in-context learning (ICL) has been recognized for its innovative ability to adapt to new tasks, relying on examples rather than retraining or fine-tuning. This paper delves into the critical issue of ICL's susceptibility to data poisoning attacks, an area not yet fully explored. We wonder whether ICL is vulnerable, with adversaries capable of manipulating example data to degrade model performance. To address this, we introduce ICLPoison, a specialized attacking framework conceived to exploit the learning mechanisms of ICL. Our approach uniquely employs discrete text perturbations to strategically influence the hidden states of LLMs during the ICL process. We outline three representative strategies to implement attacks under our framework, each rigorously evaluated across a variety of models and tasks. Our comprehensive tests, including trials on the sophisticated GPT-4 model, demonstrate that ICL's performance is significantly compromised under our framework. These revelations indicate an urgent need for enhanced defense mechanisms to safeguard the integrity and reliability of LLMs in applications relying on in-context learning.



## **47. Towards Sustainable SecureML: Quantifying Carbon Footprint of Adversarial Machine Learning**

cs.LG

Accepted at GreenNet Workshop @ IEEE International Conference on  Communications (IEEE ICC 2024)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.19009v1) [paper-pdf](http://arxiv.org/pdf/2403.19009v1)

**Authors**: Syed Mhamudul Hasan, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The widespread adoption of machine learning (ML) across various industries has raised sustainability concerns due to its substantial energy usage and carbon emissions. This issue becomes more pressing in adversarial ML, which focuses on enhancing model security against different network-based attacks. Implementing defenses in ML systems often necessitates additional computational resources and network security measures, exacerbating their environmental impacts. In this paper, we pioneer the first investigation into adversarial ML's carbon footprint, providing empirical evidence connecting greater model robustness to higher emissions. Addressing the critical need to quantify this trade-off, we introduce the Robustness Carbon Trade-off Index (RCTI). This novel metric, inspired by economic elasticity principles, captures the sensitivity of carbon emissions to changes in adversarial robustness. We demonstrate the RCTI through an experiment involving evasion attacks, analyzing the interplay between robustness against attacks, performance, and carbon emissions.



## **48. Robustness and Visual Explanation for Black Box Image, Video, and ECG Signal Classification with Reinforcement Learning**

cs.LG

AAAI Proceedings reference:  https://ojs.aaai.org/index.php/AAAI/article/view/30579

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18985v1) [paper-pdf](http://arxiv.org/pdf/2403.18985v1)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Avisek Naug, Sahand Ghorbanpour

**Abstract**: We present a generic Reinforcement Learning (RL) framework optimized for crafting adversarial attacks on different model types spanning from ECG signal analysis (1D), image classification (2D), and video classification (3D). The framework focuses on identifying sensitive regions and inducing misclassifications with minimal distortions and various distortion types. The novel RL method outperforms state-of-the-art methods for all three applications, proving its efficiency. Our RL approach produces superior localization masks, enhancing interpretability for image classification and ECG analysis models. For applications such as ECG analysis, our platform highlights critical ECG segments for clinicians while ensuring resilience against prevalent distortions. This comprehensive tool aims to bolster both resilience with adversarial training and transparency across varied applications and data types.



## **49. Deep Learning for Robust and Explainable Models in Computer Vision**

cs.CV

150 pages, 37 figures, 12 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18674v1) [paper-pdf](http://arxiv.org/pdf/2403.18674v1)

**Authors**: Mohammadreza Amirian

**Abstract**: Recent breakthroughs in machine and deep learning (ML and DL) research have provided excellent tools for leveraging enormous amounts of data and optimizing huge models with millions of parameters to obtain accurate networks for image processing. These developments open up tremendous opportunities for using artificial intelligence (AI) in the automation and human assisted AI industry. However, as more and more models are deployed and used in practice, many challenges have emerged. This thesis presents various approaches that address robustness and explainability challenges for using ML and DL in practice.   Robustness and reliability are the critical components of any model before certification and deployment in practice. Deep convolutional neural networks (CNNs) exhibit vulnerability to transformations of their inputs, such as rotation and scaling, or intentional manipulations as described in the adversarial attack literature. In addition, building trust in AI-based models requires a better understanding of current models and developing methods that are more explainable and interpretable a priori.   This thesis presents developments in computer vision models' robustness and explainability. Furthermore, this thesis offers an example of using vision models' feature response visualization (models' interpretations) to improve robustness despite interpretability and robustness being seemingly unrelated in the related research. Besides methodological developments for robust and explainable vision models, a key message of this thesis is introducing model interpretation techniques as a tool for understanding vision models and improving their design and robustness. In addition to the theoretical developments, this thesis demonstrates several applications of ML and DL in different contexts, such as medical imaging and affective computing.



## **50. LCANets++: Robust Audio Classification using Multi-layer Neural Networks with Lateral Competition**

cs.SD

Accepted at 2024 IEEE International Conference on Acoustics, Speech  and Signal Processing Workshops (ICASSPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2308.12882v2) [paper-pdf](http://arxiv.org/pdf/2308.12882v2)

**Authors**: Sayanton V. Dibbo, Juston S. Moore, Garrett T. Kenyon, Michael A. Teti

**Abstract**: Audio classification aims at recognizing audio signals, including speech commands or sound events. However, current audio classifiers are susceptible to perturbations and adversarial attacks. In addition, real-world audio classification tasks often suffer from limited labeled data. To help bridge these gaps, previous work developed neuro-inspired convolutional neural networks (CNNs) with sparse coding via the Locally Competitive Algorithm (LCA) in the first layer (i.e., LCANets) for computer vision. LCANets learn in a combination of supervised and unsupervised learning, reducing dependency on labeled samples. Motivated by the fact that auditory cortex is also sparse, we extend LCANets to audio recognition tasks and introduce LCANets++, which are CNNs that perform sparse coding in multiple layers via LCA. We demonstrate that LCANets++ are more robust than standard CNNs and LCANets against perturbations, e.g., background noise, as well as black-box and white-box attacks, e.g., evasion and fast gradient sign (FGSM) attacks.



