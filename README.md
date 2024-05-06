# Latest Adversarial Attack Papers
**update at 2024-05-06 11:09:44**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Purify Unlearnable Examples via Rate-Constrained Variational Autoencoders**

cs.CR

Accepted by ICML 2024

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01460v1) [paper-pdf](http://arxiv.org/pdf/2405.01460v1)

**Authors**: Yi Yu, Yufei Wang, Song Xia, Wenhan Yang, Shijian Lu, Yap-Peng Tan, Alex C. Kot

**Abstract**: Unlearnable examples (UEs) seek to maximize testing error by making subtle modifications to training examples that are correctly labeled. Defenses against these poisoning attacks can be categorized based on whether specific interventions are adopted during training. The first approach is training-time defense, such as adversarial training, which can mitigate poisoning effects but is computationally intensive. The other approach is pre-training purification, e.g., image short squeezing, which consists of several simple compressions but often encounters challenges in dealing with various UEs. Our work provides a novel disentanglement mechanism to build an efficient pre-training purification method. Firstly, we uncover rate-constrained variational autoencoders (VAEs), demonstrating a clear tendency to suppress the perturbations in UEs. We subsequently conduct a theoretical analysis for this phenomenon. Building upon these insights, we introduce a disentangle variational autoencoder (D-VAE), capable of disentangling the perturbations with learnable class-wise embeddings. Based on this network, a two-stage purification approach is naturally developed. The first stage focuses on roughly eliminating perturbations, while the second stage produces refined, poison-free results, ensuring effectiveness and robustness across various scenarios. Extensive experiments demonstrate the remarkable performance of our method across CIFAR-10, CIFAR-100, and a 100-class ImageNet-subset. Code is available at https://github.com/yuyi-sd/D-VAE.



## **2. Position Paper: Beyond Robustness Against Single Attack Types**

cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01349v1) [paper-pdf](http://arxiv.org/pdf/2405.01349v1)

**Authors**: Sihui Dai, Chong Xiang, Tong Wu, Prateek Mittal

**Abstract**: Current research on defending against adversarial examples focuses primarily on achieving robustness against a single attack type such as $\ell_2$ or $\ell_{\infty}$-bounded attacks. However, the space of possible perturbations is much larger and currently cannot be modeled by a single attack type. The discrepancy between the focus of current defenses and the space of attacks of interest calls to question the practicality of existing defenses and the reliability of their evaluation. In this position paper, we argue that the research community should look beyond single attack robustness, and we draw attention to three potential directions involving robustness against multiple attacks: simultaneous multiattack robustness, unforeseen attack robustness, and a newly defined problem setting which we call continual adaptive robustness. We provide a unified framework which rigorously defines these problem settings, synthesize existing research in these fields, and outline open directions. We hope that our position paper inspires more research in simultaneous multiattack, unforeseen attack, and continual adaptive robustness.



## **3. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.07308v4) [paper-pdf](http://arxiv.org/pdf/2308.07308v4)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense



## **4. Causal Influence in Federated Edge Inference**

cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01260v1) [paper-pdf](http://arxiv.org/pdf/2405.01260v1)

**Authors**: Mert Kayaalp, Yunus Inan, Visa Koivunen, Ali H. Sayed

**Abstract**: In this paper, we consider a setting where heterogeneous agents with connectivity are performing inference using unlabeled streaming data. Observed data are only partially informative about the target variable of interest. In order to overcome the uncertainty, agents cooperate with each other by exchanging their local inferences with and through a fusion center. To evaluate how each agent influences the overall decision, we adopt a causal framework in order to distinguish the actual influence of agents from mere correlations within the decision-making process. Various scenarios reflecting different agent participation patterns and fusion center policies are investigated. We derive expressions to quantify the causal impact of each agent on the joint decision, which could be beneficial for anticipating and addressing atypical scenarios, such as adversarial attacks or system malfunctions. We validate our theoretical results with numerical simulations and a real-world application of multi-camera crowd counting.



## **5. Boosting Jailbreak Attack with Momentum**

cs.LG

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01229v1) [paper-pdf](http://arxiv.org/pdf/2405.01229v1)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-documented \textit{jailbreak} attack. Recently, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous iterations. Specifically, we introduce the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which incorporates a momentum term into the gradient heuristic. Experimental results showcase the notable enhancement achieved by MAP in gradient-based attacks on aligned language models. Our code is available at https://github.com/weizeming/momentum-attack-llm.



## **6. Neural Exec: Learning (and Learning from) Execution Triggers for Prompt Injection Attacks**

cs.CR

v0.2

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2403.03792v2) [paper-pdf](http://arxiv.org/pdf/2403.03792v2)

**Authors**: Dario Pasquini, Martin Strohmeier, Carmela Troncoso

**Abstract**: We introduce a new family of prompt injection attacks, termed Neural Exec. Unlike known attacks that rely on handcrafted strings (e.g., "Ignore previous instructions and..."), we show that it is possible to conceptualize the creation of execution triggers as a differentiable search problem and use learning-based methods to autonomously generate them.   Our results demonstrate that a motivated adversary can forge triggers that are not only drastically more effective than current handcrafted ones but also exhibit inherent flexibility in shape, properties, and functionality. In this direction, we show that an attacker can design and generate Neural Execs capable of persisting through multi-stage preprocessing pipelines, such as in the case of Retrieval-Augmented Generation (RAG)-based applications. More critically, our findings show that attackers can produce triggers that deviate markedly in form and shape from any known attack, sidestepping existing blacklist-based detection and sanitation approaches.



## **7. The Perception-Robustness Tradeoff in Deterministic Image Restoration**

eess.IV

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2311.09253v2) [paper-pdf](http://arxiv.org/pdf/2311.09253v2)

**Authors**: Guy Ohayon, Tomer Michaeli, Michael Elad

**Abstract**: We study the behavior of deterministic methods for solving inverse problems in imaging. These methods are commonly designed to achieve two goals: (1) attaining high perceptual quality, and (2) generating reconstructions that are consistent with the measurements. We provide a rigorous proof that the better a predictor satisfies these two requirements, the larger its Lipschitz constant must be, regardless of the nature of the degradation involved. In particular, to approach perfect perceptual quality and perfect consistency, the Lipschitz constant of the model must grow to infinity. This implies that such methods are necessarily more susceptible to adversarial attacks. We demonstrate our theory on single image super-resolution algorithms, addressing both noisy and noiseless settings. We also show how this undesired behavior can be leveraged to explore the posterior distribution, thereby allowing the deterministic model to imitate stochastic methods.



## **8. Beyond the Bridge: Contention-Based Covert and Side Channel Attacks on Multi-GPU Interconnect**

cs.CR

Accepted to SEED 2024

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.03877v2) [paper-pdf](http://arxiv.org/pdf/2404.03877v2)

**Authors**: Yicheng Zhang, Ravan Nazaraliyev, Sankha Baran Dutta, Nael Abu-Ghazaleh, Andres Marquez, Kevin Barker

**Abstract**: High-speed interconnects, such as NVLink, are integral to modern multi-GPU systems, acting as a vital link between CPUs and GPUs. This study highlights the vulnerability of multi-GPU systems to covert and side channel attacks due to congestion on interconnects. An adversary can infer private information about a victim's activities by monitoring NVLink congestion without needing special permissions. Leveraging this insight, we develop a covert channel attack across two GPUs with a bandwidth of 45.5 kbps and a low error rate, and introduce a side channel attack enabling attackers to fingerprint applications through the shared NVLink interconnect.



## **9. MISLEAD: Manipulating Importance of Selected features for Learning Epsilon in Evasion Attack Deception**

cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.15656v2) [paper-pdf](http://arxiv.org/pdf/2404.15656v2)

**Authors**: Vidit Khazanchi, Pavan Kulkarni, Yuvaraj Govindarajulu, Manojkumar Parmar

**Abstract**: Emerging vulnerabilities in machine learning (ML) models due to adversarial attacks raise concerns about their reliability. Specifically, evasion attacks manipulate models by introducing precise perturbations to input data, causing erroneous predictions. To address this, we propose a methodology combining SHapley Additive exPlanations (SHAP) for feature importance analysis with an innovative Optimal Epsilon technique for conducting evasion attacks. Our approach begins with SHAP-based analysis to understand model vulnerabilities, crucial for devising targeted evasion strategies. The Optimal Epsilon technique, employing a Binary Search algorithm, efficiently determines the minimum epsilon needed for successful evasion. Evaluation across diverse machine learning architectures demonstrates the technique's precision in generating adversarial samples, underscoring its efficacy in manipulating model outcomes. This study emphasizes the critical importance of continuous assessment and monitoring to identify and mitigate potential security risks in machine learning systems.



## **10. Adversarial Attacks and Defense for Conversation Entailment Task**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.00289v2) [paper-pdf](http://arxiv.org/pdf/2405.00289v2)

**Authors**: Zhenning Yang, Ryan Krawec, Liang-Yuan Wu

**Abstract**: As the deployment of NLP systems in critical applications grows, ensuring the robustness of large language models (LLMs) against adversarial attacks becomes increasingly important. Large language models excel in various NLP tasks but remain vulnerable to low-cost adversarial attacks. Focusing on the domain of conversation entailment, where multi-turn dialogues serve as premises to verify hypotheses, we fine-tune a transformer model to accurately discern the truthfulness of these hypotheses. Adversaries manipulate hypotheses through synonym swapping, aiming to deceive the model into making incorrect predictions. To counteract these attacks, we implemented innovative fine-tuning techniques and introduced an embedding perturbation loss method to significantly bolster the model's robustness. Our findings not only emphasize the importance of defending against adversarial attacks in NLP but also highlight the real-world implications, suggesting that enhancing model robustness is critical for reliable NLP applications.



## **11. Pixel is a Barrier: Diffusion Models Are More Adversarially Robust Than We Think**

cs.CV

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.13320v2) [paper-pdf](http://arxiv.org/pdf/2404.13320v2)

**Authors**: Haotian Xue, Yongxin Chen

**Abstract**: Adversarial examples for diffusion models are widely used as solutions for safety concerns. By adding adversarial perturbations to personal images, attackers can not edit or imitate them easily. However, it is essential to note that all these protections target the latent diffusion model (LDMs), the adversarial examples for diffusion models in the pixel space (PDMs) are largely overlooked. This may mislead us to think that the diffusion models are vulnerable to adversarial attacks like most deep models. In this paper, we show novel findings that: even though gradient-based white-box attacks can be used to attack the LDMs, they fail to attack PDMs. This finding is supported by extensive experiments of almost a wide range of attacking methods on various PDMs and LDMs with different model structures, which means diffusion models are indeed much more robust against adversarial attacks. We also find that PDMs can be used as an off-the-shelf purifier to effectively remove the adversarial patterns that were generated on LDMs to protect the images, which means that most protection methods nowadays, to some extent, cannot protect our images from malicious attacks. We hope that our insights will inspire the community to rethink the adversarial samples for diffusion models as protection methods and move forward to more effective protection. Codes are available in https://github.com/xavihart/PDM-Pure.



## **12. Intriguing Properties of Diffusion Models: An Empirical Study of the Natural Attack Capability in Text-to-Image Generative Models**

cs.CV

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.15692v2) [paper-pdf](http://arxiv.org/pdf/2308.15692v2)

**Authors**: Takami Sato, Justin Yue, Nanze Chen, Ningfei Wang, Qi Alfred Chen

**Abstract**: Denoising probabilistic diffusion models have shown breakthrough performance to generate more photo-realistic images or human-level illustrations than the prior models such as GANs. This high image-generation capability has stimulated the creation of many downstream applications in various areas. However, we find that this technology is actually a double-edged sword: We identify a new type of attack, called the Natural Denoising Diffusion (NDD) attack based on the finding that state-of-the-art deep neural network (DNN) models still hold their prediction even if we intentionally remove their robust features, which are essential to the human visual system (HVS), through text prompts. The NDD attack shows a significantly high capability to generate low-cost, model-agnostic, and transferable adversarial attacks by exploiting the natural attack capability in diffusion models. To systematically evaluate the risk of the NDD attack, we perform a large-scale empirical study with our newly created dataset, the Natural Denoising Diffusion Attack (NDDA) dataset. We evaluate the natural attack capability by answering 6 research questions. Through a user study, we find that it can achieve an 88% detection rate while being stealthy to 93% of human subjects; we also find that the non-robust features embedded by diffusion models contribute to the natural attack capability. To confirm the model-agnostic and transferable attack capability, we perform the NDD attack against the Tesla Model 3 and find that 73% of the physically printed attacks can be detected as stop signs. Our hope is that the study and dataset can help our community be aware of the risks in diffusion models and facilitate further research toward robust DNN models.



## **13. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.07921v2) [paper-pdf](http://arxiv.org/pdf/2404.07921v2)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.



## **14. A Survey on Transferability of Adversarial Examples across Deep Neural Networks**

cs.CV

Accepted to Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2310.17626v2) [paper-pdf](http://arxiv.org/pdf/2310.17626v2)

**Authors**: Jindong Gu, Xiaojun Jia, Pau de Jorge, Wenqain Yu, Xinwei Liu, Avery Ma, Yuan Xun, Anjun Hu, Ashkan Khakzar, Zhijiang Li, Xiaochun Cao, Philip Torr

**Abstract**: The emergence of Deep Neural Networks (DNNs) has revolutionized various domains by enabling the resolution of complex tasks spanning image recognition, natural language processing, and scientific problem-solving. However, this progress has also brought to light a concerning vulnerability: adversarial examples. These crafted inputs, imperceptible to humans, can manipulate machine learning models into making erroneous predictions, raising concerns for safety-critical applications. An intriguing property of this phenomenon is the transferability of adversarial examples, where perturbations crafted for one model can deceive another, often with a different architecture. This intriguing property enables black-box attacks which circumvents the need for detailed knowledge of the target model. This survey explores the landscape of the adversarial transferability of adversarial examples. We categorize existing methodologies to enhance adversarial transferability and discuss the fundamental principles guiding each approach. While the predominant body of research primarily concentrates on image classification, we also extend our discussion to encompass other vision tasks and beyond. Challenges and opportunities are discussed, highlighting the importance of fortifying DNNs against adversarial vulnerabilities in an evolving landscape.



## **15. Why You Should Not Trust Interpretations in Machine Learning: Adversarial Attacks on Partial Dependence Plots**

cs.LG

**SubmitDate**: 2024-05-01    [abs](http://arxiv.org/abs/2404.18702v2) [paper-pdf](http://arxiv.org/pdf/2404.18702v2)

**Authors**: Xi Xin, Giles Hooker, Fei Huang

**Abstract**: The adoption of artificial intelligence (AI) across industries has led to the widespread use of complex black-box models and interpretation tools for decision making. This paper proposes an adversarial framework to uncover the vulnerability of permutation-based interpretation methods for machine learning tasks, with a particular focus on partial dependence (PD) plots. This adversarial framework modifies the original black box model to manipulate its predictions for instances in the extrapolation domain. As a result, it produces deceptive PD plots that can conceal discriminatory behaviors while preserving most of the original model's predictions. This framework can produce multiple fooled PD plots via a single model. By using real-world datasets including an auto insurance claims dataset and COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) dataset, our results show that it is possible to intentionally hide the discriminatory behavior of a predictor and make the black-box model appear neutral through interpretation tools like PD plots while retaining almost all the predictions of the original black-box model. Managerial insights for regulators and practitioners are provided based on the findings.



## **16. Certified Adversarial Robustness of Machine Learning-based Malware Detectors via (De)Randomized Smoothing**

cs.CR

**SubmitDate**: 2024-05-01    [abs](http://arxiv.org/abs/2405.00392v1) [paper-pdf](http://arxiv.org/pdf/2405.00392v1)

**Authors**: Daniel Gibert, Luca Demetrio, Giulio Zizzo, Quan Le, Jordi Planes, Battista Biggio

**Abstract**: Deep learning-based malware detection systems are vulnerable to adversarial EXEmples - carefully-crafted malicious programs that evade detection with minimal perturbation. As such, the community is dedicating effort to develop mechanisms to defend against adversarial EXEmples. However, current randomized smoothing-based defenses are still vulnerable to attacks that inject blocks of adversarial content. In this paper, we introduce a certifiable defense against patch attacks that guarantees, for a given executable and an adversarial patch size, no adversarial EXEmple exist. Our method is inspired by (de)randomized smoothing which provides deterministic robustness certificates. During training, a base classifier is trained using subsets of continguous bytes. At inference time, our defense splits the executable into non-overlapping chunks, classifies each chunk independently, and computes the final prediction through majority voting to minimize the influence of injected content. Furthermore, we introduce a preprocessing step that fixes the size of the sections and headers to a multiple of the chunk size. As a consequence, the injected content is confined to an integer number of chunks without tampering the other chunks containing the real bytes of the input examples, allowing us to extend our certified robustness guarantees to content insertion attacks. We perform an extensive ablation study, by comparing our defense with randomized smoothing-based defenses against a plethora of content manipulation attacks and neural network architectures. Results show that our method exhibits unmatched robustness against strong content-insertion attacks, outperforming randomized smoothing-based defenses in the literature.



## **17. Graphene: Infrastructure Security Posture Analysis with AI-generated Attack Graphs**

cs.CR

**SubmitDate**: 2024-05-01    [abs](http://arxiv.org/abs/2312.13119v2) [paper-pdf](http://arxiv.org/pdf/2312.13119v2)

**Authors**: Xin Jin, Charalampos Katsis, Fan Sang, Jiahao Sun, Elisa Bertino, Ramana Rao Kompella, Ashish Kundu

**Abstract**: The rampant occurrence of cybersecurity breaches imposes substantial limitations on the progress of network infrastructures, leading to compromised data, financial losses, potential harm to individuals, and disruptions in essential services. The current security landscape demands the urgent development of a holistic security assessment solution that encompasses vulnerability analysis and investigates the potential exploitation of these vulnerabilities as attack paths. In this paper, we propose Graphene, an advanced system designed to provide a detailed analysis of the security posture of computing infrastructures. Using user-provided information, such as device details and software versions, Graphene performs a comprehensive security assessment. This assessment includes identifying associated vulnerabilities and constructing potential attack graphs that adversaries can exploit. Furthermore, Graphene evaluates the exploitability of these attack paths and quantifies the overall security posture through a scoring mechanism. The system takes a holistic approach by analyzing security layers encompassing hardware, system, network, and cryptography. Furthermore, Graphene delves into the interconnections between these layers, exploring how vulnerabilities in one layer can be leveraged to exploit vulnerabilities in others. In this paper, we present the end-to-end pipeline implemented in Graphene, showcasing the systematic approach adopted for conducting this thorough security analysis.



## **18. SimAC: A Simple Anti-Customization Method for Protecting Face Privacy against Text-to-Image Synthesis of Diffusion Models**

cs.CV

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2312.07865v2) [paper-pdf](http://arxiv.org/pdf/2312.07865v2)

**Authors**: Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, Qidong Huang

**Abstract**: Despite the success of diffusion-based customization methods on visual content creation, increasing concerns have been raised about such techniques from both privacy and political perspectives. To tackle this issue, several anti-customization methods have been proposed in very recent months, predominantly grounded in adversarial attacks. Unfortunately, most of these methods adopt straightforward designs, such as end-to-end optimization with a focus on adversarially maximizing the original training loss, thereby neglecting nuanced internal properties intrinsic to the diffusion model, and even leading to ineffective optimization in some diffusion time steps.In this paper, we strive to bridge this gap by undertaking a comprehensive exploration of these inherent properties, to boost the performance of current anti-customization approaches. Two aspects of properties are investigated: 1) We examine the relationship between time step selection and the model's perception in the frequency domain of images and find that lower time steps can give much more contributions to adversarial noises. This inspires us to propose an adaptive greedy search for optimal time steps that seamlessly integrates with existing anti-customization methods. 2) We scrutinize the roles of features at different layers during denoising and devise a sophisticated feature-based optimization framework for anti-customization.Experiments on facial benchmarks demonstrate that our approach significantly increases identity disruption, thereby protecting user privacy and copyright. Our code is available at: https://github.com/somuchtome/SimAC.



## **19. Adversarial Example Soups: Improving Transferability and Stealthiness for Free**

cs.CV

Under review

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2402.18370v2) [paper-pdf](http://arxiv.org/pdf/2402.18370v2)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yulong Yang, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: Transferable adversarial examples cause practical security risks since they can mislead a target model without knowing its internal knowledge. A conventional recipe for maximizing transferability is to keep only the optimal adversarial example from all those obtained in the optimization pipeline. In this paper, for the first time, we question this convention and demonstrate that those discarded, sub-optimal adversarial examples can be reused to boost transferability. Specifically, we propose ``Adversarial Example Soups'' (AES), with AES-tune for averaging discarded adversarial examples in hyperparameter tuning and AES-rand for stability testing. In addition, our AES is inspired by ``model soups'', which averages weights of multiple fine-tuned models for improved accuracy without increasing inference time. Extensive experiments validate the global effectiveness of our AES, boosting 10 state-of-the-art transfer attacks and their combinations by up to 13% against 10 diverse (defensive) target models. We also show the possibility of generalizing AES to other types, e.g., directly averaging multiple in-the-wild adversarial examples that yield comparable success. A promising byproduct of AES is the improved stealthiness of adversarial examples since the perturbation variances are naturally reduced.



## **20. Causal Perception Inspired Representation Learning for Trustworthy Image Quality Assessment**

cs.CV

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19567v1) [paper-pdf](http://arxiv.org/pdf/2404.19567v1)

**Authors**: Lei Wang, Desen Yuan

**Abstract**: Despite great success in modeling visual perception, deep neural network based image quality assessment (IQA) still remains unreliable in real-world applications due to its vulnerability to adversarial perturbations and the inexplicit black-box structure. In this paper, we propose to build a trustworthy IQA model via Causal Perception inspired Representation Learning (CPRL), and a score reflection attack method for IQA model. More specifically, we assume that each image is composed of Causal Perception Representation (CPR) and non-causal perception representation (N-CPR). CPR serves as the causation of the subjective quality label, which is invariant to the imperceptible adversarial perturbations. Inversely, N-CPR presents spurious associations with the subjective quality label, which may significantly change with the adversarial perturbations. To extract the CPR from each input image, we develop a soft ranking based channel-wise activation function to mediate the causally sufficient (beneficial for high prediction accuracy) and necessary (beneficial for high robustness) deep features, and based on intervention employ minimax game to optimize. Experiments on four benchmark databases show that the proposed CPRL method outperforms many state-of-the-art adversarial defense methods and provides explicit model interpretation.



## **21. AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples**

cs.LG

https://attackbench.github.io

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19460v1) [paper-pdf](http://arxiv.org/pdf/2404.19460v1)

**Authors**: Antonio Emanuele Cinà, Jérôme Rony, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Ismail Ben Ayed, Fabio Roli

**Abstract**: Adversarial examples are typically optimized with gradient-based attacks. While novel attacks are continuously proposed, each is shown to outperform its predecessors using different experimental setups, hyperparameter settings, and number of forward and backward calls to the target models. This provides overly-optimistic and even biased evaluations that may unfairly favor one particular attack over the others. In this work, we aim to overcome these limitations by proposing AttackBench, i.e., the first evaluation framework that enables a fair comparison among different attacks. To this end, we first propose a categorization of gradient-based attacks, identifying their main components and differences. We then introduce our framework, which evaluates their effectiveness and efficiency. We measure these characteristics by (i) defining an optimality metric that quantifies how close an attack is to the optimal solution, and (ii) limiting the number of forward and backward queries to the model, such that all attacks are compared within a given maximum query budget. Our extensive experimental analysis compares more than 100 attack implementations with a total of over 800 different configurations against CIFAR-10 and ImageNet models, highlighting that only very few attacks outperform all the competing approaches. Within this analysis, we shed light on several implementation issues that prevent many attacks from finding better solutions or running at all. We release AttackBench as a publicly available benchmark, aiming to continuously update it to include and evaluate novel gradient-based attacks for optimizing adversarial examples.



## **22. Probing Unlearned Diffusion Models: A Transferable Adversarial Attack Perspective**

cs.CV

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19382v1) [paper-pdf](http://arxiv.org/pdf/2404.19382v1)

**Authors**: Xiaoxuan Han, Songlin Yang, Wei Wang, Yang Li, Jing Dong

**Abstract**: Advanced text-to-image diffusion models raise safety concerns regarding identity privacy violation, copyright infringement, and Not Safe For Work content generation. Towards this, unlearning methods have been developed to erase these involved concepts from diffusion models. However, these unlearning methods only shift the text-to-image mapping and preserve the visual content within the generative space of diffusion models, leaving a fatal flaw for restoring these erased concepts. This erasure trustworthiness problem needs probe, but previous methods are sub-optimal from two perspectives: (1) Lack of transferability: Some methods operate within a white-box setting, requiring access to the unlearned model. And the learned adversarial input often fails to transfer to other unlearned models for concept restoration; (2) Limited attack: The prompt-level methods struggle to restore narrow concepts from unlearned models, such as celebrity identity. Therefore, this paper aims to leverage the transferability of the adversarial attack to probe the unlearning robustness under a black-box setting. This challenging scenario assumes that the unlearning method is unknown and the unlearned model is inaccessible for optimization, requiring the attack to be capable of transferring across different unlearned models. Specifically, we employ an adversarial search strategy to search for the adversarial embedding which can transfer across different unlearned models. This strategy adopts the original Stable Diffusion model as a surrogate model to iteratively erase and search for embeddings, enabling it to find the embedding that can restore the target concept for different unlearning methods. Extensive experiments demonstrate the transferability of the searched adversarial embedding across several state-of-the-art unlearning methods and its effectiveness for different levels of concepts.



## **23. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19287v1) [paper-pdf](http://arxiv.org/pdf/2404.19287v1)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.



## **24. Proof-of-Learning with Incentive Security**

cs.CR

17 pages, 5 figures

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.09005v3) [paper-pdf](http://arxiv.org/pdf/2404.09005v3)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.



## **25. Illusory Attacks: Detectability Matters in Adversarial Attacks on Sequential Decision-Makers**

cs.AI

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2207.10170v4) [paper-pdf](http://arxiv.org/pdf/2207.10170v4)

**Authors**: Tim Franzmeyer, Stephen McAleer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstract**: Autonomous agents deployed in the real world need to be robust against adversarial attacks on sensory inputs. Robustifying agent policies requires anticipating the strongest attacks possible. We demonstrate that existing observation-space attacks on reinforcement learning agents have a common weakness: while effective, their lack of information-theoretic detectability constraints makes them detectable using automated means or human inspection. Detectability is undesirable to adversaries as it may trigger security escalations. We introduce \eattacks{}, a novel form of adversarial attack on sequential decision-makers that is both effective and of $\epsilon$-bounded statistical detectability. We propose a novel dual ascent algorithm to learn such attacks end-to-end. Compared to existing attacks, we empirically find \eattacks{} to be significantly harder to detect with automated methods, and a small study with human participants (IRB approval under reference R84123/RE001) suggests they are similarly harder to detect for humans. Our findings suggest the need for better anomaly detectors, as well as effective hardware- and system-level defenses. The project website can be found at https://tinyurl.com/illusory-attacks.



## **26. Certification of Speaker Recognition Models to Additive Perturbations**

cs.SD

9 pages, 9 figures

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18791v1) [paper-pdf](http://arxiv.org/pdf/2404.18791v1)

**Authors**: Dmitrii Korzh, Elvir Karimov, Mikhail Pautov, Oleg Y. Rogov, Ivan Oseledets

**Abstract**: Speaker recognition technology is applied in various tasks ranging from personal virtual assistants to secure access systems. However, the robustness of these systems against adversarial attacks, particularly to additive perturbations, remains a significant challenge. In this paper, we pioneer applying robustness certification techniques to speaker recognition, originally developed for the image domain. In our work, we cover this gap by transferring and improving randomized smoothing certification techniques against norm-bounded additive perturbations for classification and few-shot learning tasks to speaker recognition. We demonstrate the effectiveness of these methods on VoxCeleb 1 and 2 datasets for several models. We expect this work to improve voice-biometry robustness, establish a new certification benchmark, and accelerate research of certification methods in the audio domain.



## **27. Universal Jailbreak Backdoors from Poisoned Human Feedback**

cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2311.14455v4) [paper-pdf](http://arxiv.org/pdf/2311.14455v4)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.



## **28. Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection**

cs.CV

Accepted for publication, 3rd ACM Int. Workshop on Multimedia AI  against Disinformation (MAD'24) at ACM ICMR'24, June 10, 2024, Phuket,  Thailand. This is the "accepted version"

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18649v1) [paper-pdf](http://arxiv.org/pdf/2404.18649v1)

**Authors**: Konstantinos Tsigos, Evlampios Apostolidis, Spyridon Baxevanakis, Symeon Papadopoulos, Vasileios Mezaris

**Abstract**: In this paper we propose a new framework for evaluating the performance of explanation methods on the decisions of a deepfake detector. This framework assesses the ability of an explanation method to spot the regions of a fake image with the biggest influence on the decision of the deepfake detector, by examining the extent to which these regions can be modified through a set of adversarial attacks, in order to flip the detector's prediction or reduce its initial prediction; we anticipate a larger drop in deepfake detection accuracy and prediction, for methods that spot these regions more accurately. Based on this framework, we conduct a comparative study using a state-of-the-art model for deepfake detection that has been trained on the FaceForensics++ dataset, and five explanation methods from the literature. The findings of our quantitative and qualitative evaluations document the advanced performance of the LIME explanation method against the other compared ones, and indicate this method as the most appropriate for explaining the decisions of the utilized deepfake detector.



## **29. Assessing Cybersecurity Vulnerabilities in Code Large Language Models**

cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18567v1) [paper-pdf](http://arxiv.org/pdf/2404.18567v1)

**Authors**: Md Imran Hossen, Jianyi Zhang, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Code Large Language Models (Code LLMs) are increasingly utilized as AI coding assistants and integrated into various applications. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. To bridge this gap, this paper presents EvilInstructCoder, a framework specifically designed to assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs to adversarial attacks. EvilInstructCoder introduces the Adversarial Code Injection Engine to automatically generate malicious code snippets and inject them into benign code to poison instruction tuning datasets. It incorporates practical threat models to reflect real-world adversaries with varying capabilities and evaluates the exploitability of instruction-tuned Code LLMs under these diverse adversarial attack scenarios. Through the use of EvilInstructCoder, we conduct a comprehensive investigation into the exploitability of instruction tuning for coding tasks using three state-of-the-art Code LLM models: CodeLlama, DeepSeek-Coder, and StarCoder2, under various adversarial attack scenarios. Our experimental results reveal a significant vulnerability in these models, demonstrating that adversaries can manipulate the models to generate malicious payloads within benign code contexts in response to natural language instructions. For instance, under the backdoor attack setting, by poisoning only 81 samples (0.5\% of the entire instruction dataset), we achieve Attack Success Rate at 1 (ASR@1) scores ranging from 76\% to 86\% for different model families. Our study sheds light on the critical cybersecurity vulnerabilities posed by instruction-tuned Code LLMs and emphasizes the urgent necessity for robust defense mechanisms to mitigate the identified vulnerabilities.



## **30. Machine Learning for Windows Malware Detection and Classification: Methods, Challenges and Ongoing Research**

cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18541v1) [paper-pdf](http://arxiv.org/pdf/2404.18541v1)

**Authors**: Daniel Gibert

**Abstract**: In this chapter, readers will explore how machine learning has been applied to build malware detection systems designed for the Windows operating system. This chapter starts by introducing the main components of a Machine Learning pipeline, highlighting the challenges of collecting and maintaining up-to-date datasets. Following this introduction, various state-of-the-art malware detectors are presented, encompassing both feature-based and deep learning-based detectors. Subsequent sections introduce the primary challenges encountered by machine learning-based malware detectors, including concept drift and adversarial attacks. Lastly, this chapter concludes by providing a brief overview of the ongoing research on adversarial defenses.



## **31. A Systematic Evaluation of Adversarial Attacks against Speech Emotion Recognition Models**

cs.SD

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18514v1) [paper-pdf](http://arxiv.org/pdf/2404.18514v1)

**Authors**: Nicolas Facchinetti, Federico Simonetta, Stavros Ntalampiras

**Abstract**: Speech emotion recognition (SER) is constantly gaining attention in recent years due to its potential applications in diverse fields and thanks to the possibility offered by deep learning technologies. However, recent studies have shown that deep learning models can be vulnerable to adversarial attacks. In this paper, we systematically assess this problem by examining the impact of various adversarial white-box and black-box attacks on different languages and genders within the context of SER. We first propose a suitable methodology for audio data processing, feature extraction, and CNN-LSTM architecture. The observed outcomes highlighted the significant vulnerability of CNN-LSTM models to adversarial examples (AEs). In fact, all the considered adversarial attacks are able to significantly reduce the performance of the constructed models. Furthermore, when assessing the efficacy of the attacks, minor differences were noted between the languages analyzed as well as between male and female speech. In summary, this work contributes to the understanding of the robustness of CNN-LSTM models, particularly in SER scenarios, and the impact of AEs. Interestingly, our findings serve as a baseline for a) developing more robust algorithms for SER, b) designing more effective attacks, c) investigating possible defenses, d) improved understanding of the vocal differences between different languages and genders, and e) overall, enhancing our comprehension of the SER task.



## **32. PriSampler: Mitigating Property Inference of Diffusion Models**

cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2306.05208v2) [paper-pdf](http://arxiv.org/pdf/2306.05208v2)

**Authors**: Hailong Hu, Jun Pang

**Abstract**: Diffusion models have been remarkably successful in data synthesis. However, when these models are applied to sensitive datasets, such as banking and human face data, they might bring up severe privacy concerns. This work systematically presents the first privacy study about property inference attacks against diffusion models, where adversaries aim to extract sensitive global properties of its training set from a diffusion model. Specifically, we focus on the most practical attack scenario: adversaries are restricted to accessing only synthetic data. Under this realistic scenario, we conduct a comprehensive evaluation of property inference attacks on various diffusion models trained on diverse data types, including tabular and image datasets. A broad range of evaluations reveals that diffusion models and their samplers are universally vulnerable to property inference attacks. In response, we propose a new model-agnostic plug-in method PriSampler to mitigate the risks of the property inference of diffusion models. PriSampler can be directly applied to well-trained diffusion models and support both stochastic and deterministic sampling. Extensive experiments illustrate the effectiveness of our defense, and it can lead adversaries to infer the proportion of properties as close as predefined values that model owners wish. Notably, PriSampler also shows its significantly superior performance to diffusion models trained with differential privacy on both model utility and defense performance. This work will elevate the awareness of preventing property inference attacks and encourage privacy-preserving synthetic data release.



## **33. Laccolith: Hypervisor-Based Adversary Emulation with Anti-Detection**

cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2311.08274v3) [paper-pdf](http://arxiv.org/pdf/2311.08274v3)

**Authors**: Vittorio Orbinato, Marco Carlo Feliciano, Domenico Cotroneo, Roberto Natella

**Abstract**: Advanced Persistent Threats (APTs) represent the most threatening form of attack nowadays since they can stay undetected for a long time. Adversary emulation is a proactive approach for preparing against these attacks. However, adversary emulation tools lack the anti-detection abilities of APTs. We introduce Laccolith, a hypervisor-based solution for adversary emulation with anti-detection to fill this gap. We also present an experimental study to compare Laccolith with MITRE CALDERA, a state-of-the-art solution for adversary emulation, against five popular anti-virus products. We found that CALDERA cannot evade detection, limiting the realism of emulated attacks, even when combined with a state-of-the-art anti-detection framework. Our experiments show that Laccolith can hide its activities from all the tested anti-virus products, thus making it suitable for realistic emulations.



## **34. ICMarks: A Robust Watermarking Framework for Integrated Circuit Physical Design IP Protection**

cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18407v1) [paper-pdf](http://arxiv.org/pdf/2404.18407v1)

**Authors**: Ruisi Zhang, Rachel Selina Rajarathnam, David Z. Pan, Farinaz Koushanfar

**Abstract**: Physical design watermarking on contemporary integrated circuit (IC) layout encodes signatures without considering the dense connections and design constraints, which could lead to performance degradation on the watermarked products. This paper presents ICMarks, a quality-preserving and robust watermarking framework for modern IC physical design. ICMarks embeds unique watermark signatures during the physical design's placement stage, thereby authenticating the IC layout ownership. ICMarks's novelty lies in (i) strategically identifying a region of cells to watermark with minimal impact on the layout performance and (ii) a two-level watermarking framework for augmented robustness toward potential removal and forging attacks. Extensive evaluations on benchmarks of different design objectives and sizes validate that ICMarks incurs no wirelength and timing metrics degradation, while successfully proving ownership. Furthermore, we demonstrate ICMarks is robust against two major watermarking attack categories, namely, watermark removal and forging attacks; even if the adversaries have prior knowledge of the watermarking schemes, the signatures cannot be removed without significantly undermining the layout quality.



## **35. DRAM-Profiler: An Experimental DRAM RowHammer Vulnerability Profiling Mechanism**

cs.CR

6 pages, 6 figures

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18396v1) [paper-pdf](http://arxiv.org/pdf/2404.18396v1)

**Authors**: Ranyang Zhou, Jacqueline T. Liu, Nakul Kochar, Sabbir Ahmed, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: RowHammer stands out as a prominent example, potentially the pioneering one, showcasing how a failure mechanism at the circuit level can give rise to a significant and pervasive security vulnerability within systems. Prior research has approached RowHammer attacks within a static threat model framework. Nonetheless, it warrants consideration within a more nuanced and dynamic model. This paper presents a low-overhead DRAM RowHammer vulnerability profiling technique termed DRAM-Profiler, which utilizes innovative test vectors for categorizing memory cells into distinct security levels. The proposed test vectors intentionally weaken the spatial correlation between the aggressors and victim rows before an attack for evaluation, thus aiding designers in mitigating RowHammer vulnerabilities in the mapping phase. While there has been no previous research showcasing the impact of such profiling to our knowledge, our study methodically assesses 128 commercial DDR4 DRAM products. The results uncover the significant variability among chips from different manufacturers in the type and quantity of RowHammer attacks that can be exploited by adversaries.



## **36. A Survey on Intermediate Fusion Methods for Collaborative Perception Categorized by Real World Challenges**

cs.CV

8 pages, 6 tables

**SubmitDate**: 2024-04-28    [abs](http://arxiv.org/abs/2404.16139v2) [paper-pdf](http://arxiv.org/pdf/2404.16139v2)

**Authors**: Melih Yazgan, Thomas Graf, Min Liu, Tobias Fleck, J. Marius Zoellner

**Abstract**: This survey analyzes intermediate fusion methods in collaborative perception for autonomous driving, categorized by real-world challenges. We examine various methods, detailing their features and the evaluation metrics they employ. The focus is on addressing challenges like transmission efficiency, localization errors, communication disruptions, and heterogeneity. Moreover, we explore strategies to counter adversarial attacks and defenses, as well as approaches to adapt to domain shifts. The objective is to present an overview of how intermediate fusion methods effectively meet these diverse challenges, highlighting their role in advancing the field of collaborative perception in autonomous driving.



## **37. Attack on Scene Flow using Point Clouds**

cs.CV

**SubmitDate**: 2024-04-28    [abs](http://arxiv.org/abs/2404.13621v2) [paper-pdf](http://arxiv.org/pdf/2404.13621v2)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. Robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants show a higher vulnerability for the optical flow networks.



## **38. Privacy-Preserving, Dropout-Resilient Aggregation in Decentralized Learning**

cs.CR

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17984v1) [paper-pdf](http://arxiv.org/pdf/2404.17984v1)

**Authors**: Ali Reza Ghavamipour, Benjamin Zi Hao Zhao, Fatih Turkmen

**Abstract**: Decentralized learning (DL) offers a novel paradigm in machine learning by distributing training across clients without central aggregation, enhancing scalability and efficiency. However, DL's peer-to-peer model raises challenges in protecting against inference attacks and privacy leaks. By forgoing central bottlenecks, DL demands privacy-preserving aggregation methods to protect data from 'honest but curious' clients and adversaries, maintaining network-wide privacy. Privacy-preserving DL faces the additional hurdle of client dropout, clients not submitting updates due to connectivity problems or unavailability, further complicating aggregation.   This work proposes three secret sharing-based dropout resilience approaches for privacy-preserving DL. Our study evaluates the efficiency, performance, and accuracy of these protocols through experiments on datasets such as MNIST, Fashion-MNIST, SVHN, and CIFAR-10. We compare our protocols with traditional secret-sharing solutions across scenarios, including those with up to 1000 clients. Evaluations show that our protocols significantly outperform conventional methods, especially in scenarios with up to 30% of clients dropout and model sizes of up to $10^6$ parameters. Our approaches demonstrate markedly high efficiency with larger models, higher dropout rates, and extensive client networks, highlighting their effectiveness in enhancing decentralized learning systems' privacy and dropout robustness.



## **39. Privacy-Preserving Aggregation for Decentralized Learning with Byzantine-Robustness**

cs.CR

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17970v1) [paper-pdf](http://arxiv.org/pdf/2404.17970v1)

**Authors**: Ali Reza Ghavamipour, Benjamin Zi Hao Zhao, Oguzhan Ersoy, Fatih Turkmen

**Abstract**: Decentralized machine learning (DL) has been receiving an increasing interest recently due to the elimination of a single point of failure, present in Federated learning setting. Yet, it is threatened by the looming threat of Byzantine clients who intentionally disrupt the learning process by broadcasting arbitrary model updates to other clients, seeking to degrade the performance of the global model. In response, robust aggregation schemes have emerged as promising solutions to defend against such Byzantine clients, thereby enhancing the robustness of Decentralized Learning. Defenses against Byzantine adversaries, however, typically require access to the updates of other clients, a counterproductive privacy trade-off that in turn increases the risk of inference attacks on those same model updates.   In this paper, we introduce SecureDL, a novel DL protocol designed to enhance the security and privacy of DL against Byzantine threats. SecureDL~facilitates a collaborative defense, while protecting the privacy of clients' model updates through secure multiparty computation. The protocol employs efficient computation of cosine similarity and normalization of updates to robustly detect and exclude model updates detrimental to model convergence. By using MNIST, Fashion-MNIST, SVHN and CIFAR-10 datasets, we evaluated SecureDL against various Byzantine attacks and compared its effectiveness with four existing defense mechanisms. Our experiments show that SecureDL is effective even in the case of attacks by the malicious majority (e.g., 80% Byzantine clients) while preserving high training accuracy.



## **40. Bounding the Expected Robustness of Graph Neural Networks Subject to Node Feature Attacks**

cs.LG

Accepted at ICLR 2024

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17947v1) [paper-pdf](http://arxiv.org/pdf/2404.17947v1)

**Authors**: Yassine Abbahaddou, Sofiane Ennadir, Johannes F. Lutzeyer, Michalis Vazirgiannis, Henrik Boström

**Abstract**: Graph Neural Networks (GNNs) have demonstrated state-of-the-art performance in various graph representation learning tasks. Recently, studies revealed their vulnerability to adversarial attacks. In this work, we theoretically define the concept of expected robustness in the context of attributed graphs and relate it to the classical definition of adversarial robustness in the graph representation learning literature. Our definition allows us to derive an upper bound of the expected robustness of Graph Convolutional Networks (GCNs) and Graph Isomorphism Networks subject to node feature attacks. Building on these findings, we connect the expected robustness of GNNs to the orthonormality of their weight matrices and consequently propose an attack-independent, more robust variant of the GCN, called the Graph Convolutional Orthonormal Robust Networks (GCORNs). We further introduce a probabilistic method to estimate the expected robustness, which allows us to evaluate the effectiveness of GCORN on several real-world datasets. Experimental experiments showed that GCORN outperforms available defense methods. Our code is publicly available at: \href{https://github.com/Sennadir/GCORN}{https://github.com/Sennadir/GCORN}.



## **41. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

cs.DC

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.14250v3) [paper-pdf](http://arxiv.org/pdf/2404.14250v3)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.



## **42. Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library**

cs.IR

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17844v1) [paper-pdf](http://arxiv.org/pdf/2404.17844v1)

**Authors**: Lei Cheng, Xiaowen Huang, Jitao Sang, Jian Yu

**Abstract**: Recently, recommender system has achieved significant success. However, due to the openness of recommender systems, they remain vulnerable to malicious attacks. Additionally, natural noise in training data and issues such as data sparsity can also degrade the performance of recommender systems. Therefore, enhancing the robustness of recommender systems has become an increasingly important research topic. In this survey, we provide a comprehensive overview of the robustness of recommender systems. Based on our investigation, we categorize the robustness of recommender systems into adversarial robustness and non-adversarial robustness. In the adversarial robustness, we introduce the fundamental principles and classical methods of recommender system adversarial attacks and defenses. In the non-adversarial robustness, we analyze non-adversarial robustness from the perspectives of data sparsity, natural noise, and data imbalance. Additionally, we summarize commonly used datasets and evaluation metrics for evaluating the robustness of recommender systems. Finally, we also discuss the current challenges in the field of recommender system robustness and potential future research directions. Additionally, to facilitate fair and efficient evaluation of attack and defense methods in adversarial robustness, we propose an adversarial robustness evaluation library--ShillingREC, and we conduct evaluations of basic attack models and recommendation models. ShillingREC project is released at https://github.com/chengleileilei/ShillingREC.



## **43. Adversarial Examples: Generation Proposal in the Context of Facial Recognition Systems**

cs.CV

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.17760v1) [paper-pdf](http://arxiv.org/pdf/2404.17760v1)

**Authors**: Marina Fuster, Ignacio Vidaurreta

**Abstract**: In this paper we investigate the vulnerability that facial recognition systems present to adversarial examples by introducing a new methodology from the attacker perspective. The technique is based on the use of the autoencoder latent space, organized with principal component analysis. We intend to analyze the potential to craft adversarial examples suitable for both dodging and impersonation attacks, against state-of-the-art systems. Our initial hypothesis, which was not strongly favoured by the results, stated that it would be possible to separate between the "identity" and "facial expression" features to produce high-quality examples. Despite the findings not supporting it, the results sparked insights into adversarial examples generation and opened new research avenues in the area.



## **44. Attacking Bayes: On the Adversarial Robustness of Bayesian Neural Networks**

cs.LG

**SubmitDate**: 2024-04-27    [abs](http://arxiv.org/abs/2404.19640v1) [paper-pdf](http://arxiv.org/pdf/2404.19640v1)

**Authors**: Yunzhen Feng, Tim G. J. Rudner, Nikolaos Tsilivis, Julia Kempe

**Abstract**: Adversarial examples have been shown to cause neural networks to fail on a wide range of vision and language tasks, but recent work has claimed that Bayesian neural networks (BNNs) are inherently robust to adversarial perturbations. In this work, we examine this claim. To study the adversarial robustness of BNNs, we investigate whether it is possible to successfully break state-of-the-art BNN inference methods and prediction pipelines using even relatively unsophisticated attacks for three tasks: (1) label prediction under the posterior predictive mean, (2) adversarial example detection with Bayesian predictive uncertainty, and (3) semantic shift detection. We find that BNNs trained with state-of-the-art approximate inference methods, and even BNNs trained with Hamiltonian Monte Carlo, are highly susceptible to adversarial attacks. We also identify various conceptual and experimental errors in previous works that claimed inherent adversarial robustness of BNNs and conclusively demonstrate that BNNs and uncertainty-aware Bayesian prediction pipelines are not inherently robust against adversarial attacks.



## **45. Overload: Latency Attacks on Object Detection for Edge Devices**

cs.CV

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2304.05370v4) [paper-pdf](http://arxiv.org/pdf/2304.05370v4)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-rung Lee

**Abstract**: Nowadays, the deployment of deep learning-based applications is an essential task owing to the increasing demands on intelligent services. In this paper, we investigate latency attacks on deep learning applications. Unlike common adversarial attacks for misclassification, the goal of latency attacks is to increase the inference time, which may stop applications from responding to the requests within a reasonable time. This kind of attack is ubiquitous for various applications, and we use object detection to demonstrate how such kind of attacks work. We also design a framework named Overload to generate latency attacks at scale. Our method is based on a newly formulated optimization problem and a novel technique, called spatial attention. This attack serves to escalate the required computing costs during the inference time, consequently leading to an extended inference time for object detection. It presents a significant threat, especially to systems with limited computing resources. We conducted experiments using YOLOv5 models on Nvidia NX. Compared to existing methods, our method is simpler and more effective. The experimental results show that with latency attacks, the inference time of a single image can be increased ten times longer in reference to the normal setting. Moreover, our findings pose a potential new threat to all object detection tasks requiring non-maximum suppression (NMS), as our attack is NMS-agnostic.



## **46. Evaluations of Machine Learning Privacy Defenses are Misleading**

cs.CR

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2404.17399v1) [paper-pdf](http://arxiv.org/pdf/2404.17399v1)

**Authors**: Michael Aerni, Jie Zhang, Florian Tramèr

**Abstract**: Empirical defenses for machine learning privacy forgo the provable guarantees of differential privacy in the hope of achieving higher utility while resisting realistic adversaries. We identify severe pitfalls in existing empirical privacy evaluations (based on membership inference attacks) that result in misleading conclusions. In particular, we show that prior evaluations fail to characterize the privacy leakage of the most vulnerable samples, use weak attacks, and avoid comparisons with practical differential privacy baselines. In 5 case studies of empirical privacy defenses, we find that prior evaluations underestimate privacy leakage by an order of magnitude. Under our stronger evaluation, none of the empirical defenses we study are competitive with a properly tuned, high-utility DP-SGD baseline (with vacuous provable guarantees).



## **47. Enhancing Privacy and Security of Autonomous UAV Navigation**

cs.CR

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2404.17225v1) [paper-pdf](http://arxiv.org/pdf/2404.17225v1)

**Authors**: Vatsal Aggarwal, Arjun Ramesh Kaushik, Charanjit Jutla, Nalini Ratha

**Abstract**: Autonomous Unmanned Aerial Vehicles (UAVs) have become essential tools in defense, law enforcement, disaster response, and product delivery. These autonomous navigation systems require a wireless communication network, and of late are deep learning based. In critical scenarios such as border protection or disaster response, ensuring the secure navigation of autonomous UAVs is paramount. But, these autonomous UAVs are susceptible to adversarial attacks through the communication network or the deep learning models - eavesdropping / man-in-the-middle / membership inference / reconstruction. To address this susceptibility, we propose an innovative approach that combines Reinforcement Learning (RL) and Fully Homomorphic Encryption (FHE) for secure autonomous UAV navigation. This end-to-end secure framework is designed for real-time video feeds captured by UAV cameras and utilizes FHE to perform inference on encrypted input images. While FHE allows computations on encrypted data, certain computational operators are yet to be implemented. Convolutional neural networks, fully connected neural networks, activation functions and OpenAI Gym Library are meticulously adapted to the FHE domain to enable encrypted data processing. We demonstrate the efficacy of our proposed approach through extensive experimentation. Our proposed approach ensures security and privacy in autonomous UAV navigation with negligible loss in performance.



## **48. Time-Frequency Jointed Imperceptible Adversarial Attack to Brainprint Recognition with Deep Learning Models**

cs.CR

This work is accepted by ICME 2024

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2403.10021v2) [paper-pdf](http://arxiv.org/pdf/2403.10021v2)

**Authors**: Hangjie Yi, Yuhang Ming, Dongjun Liu, Wanzeng Kong

**Abstract**: EEG-based brainprint recognition with deep learning models has garnered much attention in biometric identification. Yet, studies have indicated vulnerability to adversarial attacks in deep learning models with EEG inputs. In this paper, we introduce a novel adversarial attack method that jointly attacks time-domain and frequency-domain EEG signals by employing wavelet transform. Different from most existing methods which only target time-domain EEG signals, our method not only takes advantage of the time-domain attack's potent adversarial strength but also benefits from the imperceptibility inherent in frequency-domain attack, achieving a better balance between attack performance and imperceptibility. Extensive experiments are conducted in both white- and grey-box scenarios and the results demonstrate that our attack method achieves state-of-the-art attack performance on three datasets and three deep-learning models. In the meanwhile, the perturbations in the signals attacked by our method are barely perceptible to the human visual system.



## **49. Toward Evaluating Robustness of Reinforcement Learning with Adversarial Policy**

cs.LG

Accepted by DSN 2024

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2305.02605v3) [paper-pdf](http://arxiv.org/pdf/2305.02605v3)

**Authors**: Xiang Zheng, Xingjun Ma, Shengjie Wang, Xinyu Wang, Chao Shen, Cong Wang

**Abstract**: Reinforcement learning agents are susceptible to evasion attacks during deployment. In single-agent environments, these attacks can occur through imperceptible perturbations injected into the inputs of the victim policy network. In multi-agent environments, an attacker can manipulate an adversarial opponent to influence the victim policy's observations indirectly. While adversarial policies offer a promising technique to craft such attacks, current methods are either sample-inefficient due to poor exploration strategies or require extra surrogate model training under the black-box assumption. To address these challenges, in this paper, we propose Intrinsically Motivated Adversarial Policy (IMAP) for efficient black-box adversarial policy learning in both single- and multi-agent environments. We formulate four types of adversarial intrinsic regularizers -- maximizing the adversarial state coverage, policy coverage, risk, or divergence -- to discover potential vulnerabilities of the victim policy in a principled way. We also present a novel bias-reduction method to balance the extrinsic objective and the adversarial intrinsic regularizers adaptively. Our experiments validate the effectiveness of the four types of adversarial intrinsic regularizers and the bias-reduction method in enhancing black-box adversarial policy learning across a variety of environments. Our IMAP successfully evades two types of defense methods, adversarial training and robust regularizer, decreasing the performance of the state-of-the-art robust WocaR-PPO agents by 34\%-54\% across four single-agent tasks. IMAP also achieves a state-of-the-art attacking success rate of 83.91\% in the multi-agent game YouShallNotPass. Our code is available at \url{https://github.com/x-zheng16/IMAP}.



## **50. SoK: On the Semantic AI Security in Autonomous Driving**

cs.CR

Project website: https://sites.google.com/view/cav-sec/pass

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2203.05314v2) [paper-pdf](http://arxiv.org/pdf/2203.05314v2)

**Authors**: Junjie Shen, Ningfei Wang, Ziwen Wan, Yunpeng Luo, Takami Sato, Zhisheng Hu, Xinyang Zhang, Shengjian Guo, Zhenyu Zhong, Kang Li, Ziming Zhao, Chunming Qiao, Qi Alfred Chen

**Abstract**: Autonomous Driving (AD) systems rely on AI components to make safety and correct driving decisions. Unfortunately, today's AI algorithms are known to be generally vulnerable to adversarial attacks. However, for such AI component-level vulnerabilities to be semantically impactful at the system level, it needs to address non-trivial semantic gaps both (1) from the system-level attack input spaces to those at AI component level, and (2) from AI component-level attack impacts to those at the system level. In this paper, we define such research space as semantic AI security as opposed to generic AI security. Over the past 5 years, increasingly more research works are performed to tackle such semantic AI security challenges in AD context, which has started to show an exponential growth trend.   In this paper, we perform the first systematization of knowledge of such growing semantic AD AI security research space. In total, we collect and analyze 53 such papers, and systematically taxonomize them based on research aspects critical for the security field. We summarize 6 most substantial scientific gaps observed based on quantitative comparisons both vertically among existing AD AI security works and horizontally with security works from closely-related domains. With these, we are able to provide insights and potential future directions not only at the design level, but also at the research goal, methodology, and community levels. To address the most critical scientific methodology-level gap, we take the initiative to develop an open-source, uniform, and extensible system-driven evaluation platform, named PASS, for the semantic AD AI security research community. We also use our implemented platform prototype to showcase the capabilities and benefits of such a platform using representative semantic AD AI attacks.



