# Latest Adversarial Attack Papers
**update at 2024-10-16 11:23:06**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11782v1) [paper-pdf](http://arxiv.org/pdf/2410.11782v1)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **2. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.20485v2) [paper-pdf](http://arxiv.org/pdf/2405.20485v2)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama, and show that they transfer to GPT-3.5 Turbo and GPT-4. Finally, we successfully conducted a Phantom attack on NVIDIA's black-box production RAG system, "Chat with RTX".



## **3. Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor**

cs.CR

Accepted by NeurIPS 2024. 32 pages, 7 figures, 28 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.16112v2) [paper-pdf](http://arxiv.org/pdf/2405.16112v2)

**Authors**: Shaokui Wei, Hongyuan Zha, Baoyuan Wu

**Abstract**: Data-poisoning backdoor attacks are serious security threats to machine learning models, where an adversary can manipulate the training dataset to inject backdoors into models. In this paper, we focus on in-training backdoor defense, aiming to train a clean model even when the dataset may be potentially poisoned. Unlike most existing methods that primarily detect and remove/unlearn suspicious samples to mitigate malicious backdoor attacks, we propose a novel defense approach called PDB (Proactive Defensive Backdoor). Specifically, PDB leverages the home-field advantage of defenders by proactively injecting a defensive backdoor into the model during training. Taking advantage of controlling the training process, the defensive backdoor is designed to suppress the malicious backdoor effectively while remaining secret to attackers. In addition, we introduce a reversible mapping to determine the defensive target label. During inference, PDB embeds a defensive trigger in the inputs and reverses the model's prediction, suppressing malicious backdoor and ensuring the model's utility on the original task. Experimental results across various datasets and models demonstrate that our approach achieves state-of-the-art defense performance against a wide range of backdoor attacks. The code is available at https://github.com/shawkui/Proactive_Defensive_Backdoor.



## **4. GSE: Group-wise Sparse and Explainable Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2311.17434v2) [paper-pdf](http://arxiv.org/pdf/2311.17434v2)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, often regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. We address this by presenting a two-phase algorithm that generates group-wise sparse attacks within semantically meaningful areas of an image. Initially, we optimize a quasinorm adversarial loss using the $1/2-$quasinorm proximal operator tailored for non-convex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2-$norm regularization applied to perturbation magnitudes. Rigorous evaluations on CIFAR-10 and ImageNet datasets demonstrate a remarkable increase in group-wise sparsity, e.g., $50.9\%$ on CIFAR-10 and $38.4\%$ on ImageNet (average case, targeted attack). This performance improvement is accompanied by significantly faster computation times, improved explainability, and a $100\%$ attack success rate.



## **5. Efficient and Effective Universal Adversarial Attack against Vision-Language Pre-training Models**

cs.CV

11 pages

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11639v1) [paper-pdf](http://arxiv.org/pdf/2410.11639v1)

**Authors**: Fan Yang, Yihao Huang, Kailong Wang, Ling Shi, Geguang Pu, Yang Liu, Haoyu Wang

**Abstract**: Vision-language pre-training (VLP) models, trained on large-scale image-text pairs, have become widely used across a variety of downstream vision-and-language (V+L) tasks. This widespread adoption raises concerns about their vulnerability to adversarial attacks. Non-universal adversarial attacks, while effective, are often impractical for real-time online applications due to their high computational demands per data instance. Recently, universal adversarial perturbations (UAPs) have been introduced as a solution, but existing generator-based UAP methods are significantly time-consuming. To overcome the limitation, we propose a direct optimization-based UAP approach, termed DO-UAP, which significantly reduces resource consumption while maintaining high attack performance. Specifically, we explore the necessity of multimodal loss design and introduce a useful data augmentation strategy. Extensive experiments conducted on three benchmark VLP datasets, six popular VLP models, and three classical downstream tasks demonstrate the efficiency and effectiveness of DO-UAP. Specifically, our approach drastically decreases the time consumption by 23-fold while achieving a better attack performance.



## **6. Information Importance-Aware Defense against Adversarial Attack for Automatic Modulation Classification:An XAI-Based Approach**

eess.SP

Accepted by WCSP 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11608v1) [paper-pdf](http://arxiv.org/pdf/2410.11608v1)

**Authors**: Jingchun Wang, Peihao Dong, Fuhui Zhou, Qihui Wu

**Abstract**: Deep learning (DL) has significantly improved automatic modulation classification (AMC) by leveraging neural networks as the feature extractor.However, as the DL-based AMC becomes increasingly widespread, it is faced with the severe secure issue from various adversarial attacks. Existing defense methods often suffer from the high computational cost, intractable parameter tuning, and insufficient robustness.This paper proposes an eXplainable artificial intelligence (XAI) defense approach, which uncovers the negative information caused by the adversarial attack through measuring the importance of input features based on the SHapley Additive exPlanations (SHAP).By properly removing the negative information in adversarial samples and then fine-tuning(FT) the model, the impact of the attacks on the classification result can be mitigated.Experimental results demonstrate that the proposed SHAP-FT improves the classification performance of the model by 15%-20% under different attack levels,which not only enhances model robustness against various attack levels but also reduces the resource consumption, validating its effectiveness in safeguarding communication networks.



## **7. RAUCA: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation**

cs.CV

12 pages. In Proceedings of the Forty-first International Conference  on Machine Learning (ICML), Vienna, Austria, July 21-27, 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2402.15853v2) [paper-pdf](http://arxiv.org/pdf/2402.15853v2)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle, resulting in suboptimal attack performance. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, Neural Renderer Plus (NRP), which can accurately project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA consistently outperforms existing methods in both simulation and real-world settings.



## **8. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11317v1) [paper-pdf](http://arxiv.org/pdf/2410.11317v1)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **9. On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning**

cs.LG

19 pages, 4 figures, 8 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.04682v2) [paper-pdf](http://arxiv.org/pdf/2410.04682v2)

**Authors**: Yongyi Su, Yushu Li, Nanqing Liu, Kui Jia, Xulei Yang, Chuan-Sheng Foo, Xun Xu

**Abstract**: Test-time adaptation (TTA) updates the model weights during the inference stage using testing data to enhance generalization. However, this practice exposes TTA to adversarial risks. Existing studies have shown that when TTA is updated with crafted adversarial test samples, also known as test-time poisoned data, the performance on benign samples can deteriorate. Nonetheless, the perceived adversarial risk may be overstated if the poisoned data is generated under overly strong assumptions. In this work, we first review realistic assumptions for test-time data poisoning, including white-box versus grey-box attacks, access to benign data, attack budget, and more. We then propose an effective and realistic attack method that better produces poisoned samples without access to benign samples, and derive an effective in-distribution attack objective. We also design two TTA-aware attack objectives. Our benchmarks of existing attack methods reveal that the TTA methods are more robust than previously believed. In addition, we analyze effective defense strategies to help develop adversarially robust TTA methods.



## **10. BRC20 Pinning Attack**

cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11295v1) [paper-pdf](http://arxiv.org/pdf/2410.11295v1)

**Authors**: Minfeng Qi, Qin Wang, Zhipeng Wang, Lin Zhong, Tianqing Zhu, Shiping Chen, William Knottenbelt

**Abstract**: BRC20 tokens are a type of non-fungible asset on the Bitcoin network. They allow users to embed customized content within Bitcoin satoshis. The related token frenzy has reached a market size of USD 3,650b over the past year (2023Q3-2024Q3). However, this intuitive design has not undergone serious security scrutiny.   We present the first in-depth analysis of the BRC20 transfer mechanism and identify a critical attack vector. A typical BRC20 transfer involves two bundled on-chain transactions with different fee levels: the first (i.e., Tx1) with a lower fee inscribes the transfer request, while the second (i.e., Tx2) with a higher fee finalizes the actual transfer. We find that an adversary can exploit this by sending a manipulated fee transaction (falling between the two fee levels), which allows Tx1 to be processed while Tx2 remains pinned in the mempool. This locks the BRC20 liquidity and disrupts normal transfers for users. We term this BRC20 pinning attack.   Our attack exposes an inherent design flaw that can be applied to 90+% inscription-based tokens within the Bitcoin ecosystem.   We also conducted the attack on Binance's ORDI hot wallet (the most prevalent BRC20 token and the most active wallet), resulting in a temporary suspension of ORDI withdrawals on Binance for 3.5 hours, which were shortly resumed after our communication.



## **11. Cognitive Overload Attack:Prompt Injection for Long Context**

cs.CL

40 pages, 31 Figures

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11272v1) [paper-pdf](http://arxiv.org/pdf/2410.11272v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Amin Karbasi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in performing tasks across various domains without needing explicit retraining. This capability, known as In-Context Learning (ICL), while impressive, exposes LLMs to a variety of adversarial prompts and jailbreaks that manipulate safety-trained LLMs into generating undesired or harmful output. In this paper, we propose a novel interpretation of ICL in LLMs through the lens of cognitive neuroscience, by drawing parallels between learning in human cognition with ICL. We applied the principles of Cognitive Load Theory in LLMs and empirically validate that similar to human cognition, LLMs also suffer from cognitive overload a state where the demand on cognitive processing exceeds the available capacity of the model, leading to potential errors. Furthermore, we demonstrated how an attacker can exploit ICL to jailbreak LLMs through deliberately designed prompts that induce cognitive overload on LLMs, thereby compromising the safety mechanisms of LLMs. We empirically validate this threat model by crafting various cognitive overload prompts and show that advanced models such as GPT-4, Claude-3.5 Sonnet, Claude-3 OPUS, Llama-3-70B-Instruct, Gemini-1.0-Pro, and Gemini-1.5-Pro can be successfully jailbroken, with attack success rates of up to 99.99%. Our findings highlight critical vulnerabilities in LLMs and underscore the urgency of developing robust safeguards. We propose integrating insights from cognitive load theory into the design and evaluation of LLMs to better anticipate and mitigate the risks of adversarial attacks. By expanding our experiments to encompass a broader range of models and by highlighting vulnerabilities in LLMs' ICL, we aim to ensure the development of safer and more reliable AI systems.



## **12. Adversarially Guided Stateful Defense Against Backdoor Attacks in Federated Deep Learning**

cs.LG

16 pages, Accepted at ACSAC 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11205v1) [paper-pdf](http://arxiv.org/pdf/2410.11205v1)

**Authors**: Hassan Ali, Surya Nepal, Salil S. Kanhere, Sanjay Jha

**Abstract**: Recent works have shown that Federated Learning (FL) is vulnerable to backdoor attacks. Existing defenses cluster submitted updates from clients and select the best cluster for aggregation. However, they often rely on unrealistic assumptions regarding client submissions and sampled clients population while choosing the best cluster. We show that in realistic FL settings, state-of-the-art (SOTA) defenses struggle to perform well against backdoor attacks in FL. To address this, we highlight that backdoored submissions are adversarially biased and overconfident compared to clean submissions. We, therefore, propose an Adversarially Guided Stateful Defense (AGSD) against backdoor attacks on Deep Neural Networks (DNNs) in FL scenarios. AGSD employs adversarial perturbations to a small held-out dataset to compute a novel metric, called the trust index, that guides the cluster selection without relying on any unrealistic assumptions regarding client submissions. Moreover, AGSD maintains a trust state history of each client that adaptively penalizes backdoored clients and rewards clean clients. In realistic FL settings, where SOTA defenses mostly fail to resist attacks, AGSD mostly outperforms all SOTA defenses with minimal drop in clean accuracy (5% in the worst-case compared to best accuracy) even when (a) given a very small held-out dataset -- typically AGSD assumes 50 samples (<= 0.1% of the training data) and (b) no heldout dataset is available, and out-of-distribution data is used instead. For reproducibility, our code will be openly available at: https://github.com/hassanalikhatim/AGSD.



## **13. Fast Second-Order Online Kernel Learning through Incremental Matrix Sketching and Decomposition**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11188v1) [paper-pdf](http://arxiv.org/pdf/2410.11188v1)

**Authors**: Dongxie Wen, Xiao Zhang, Zhewei Wei

**Abstract**: Online Kernel Learning (OKL) has attracted considerable research interest due to its promising predictive performance in streaming environments. Second-order approaches are particularly appealing for OKL as they often offer substantial improvements in regret guarantees. However, existing second-order OKL approaches suffer from at least quadratic time complexity with respect to the pre-set budget, rendering them unsuitable for meeting the real-time demands of large-scale streaming recommender systems. The singular value decomposition required to obtain explicit feature mapping is also computationally expensive due to the complete decomposition process. Moreover, the absence of incremental updates to manage approximate kernel space causes these algorithms to perform poorly in adversarial environments and real-world streaming recommendation datasets. To address these issues, we propose FORKS, a fast incremental matrix sketching and decomposition approach tailored for second-order OKL. FORKS constructs an incremental maintenance paradigm for second-order kernelized gradient descent, which includes incremental matrix sketching for kernel approximation and incremental matrix decomposition for explicit feature mapping construction. Theoretical analysis demonstrates that FORKS achieves a logarithmic regret guarantee on par with other second-order approaches while maintaining a linear time complexity w.r.t. the budget, significantly enhancing efficiency over existing approaches. We validate the performance of FORKS through extensive experiments conducted on real-world streaming recommendation datasets, demonstrating its superior scalability and robustness against adversarial attacks.



## **14. Sensor Deprivation Attacks for Stealthy UAV Manipulation**

cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.11131v1) [paper-pdf](http://arxiv.org/pdf/2410.11131v1)

**Authors**: Alessandro Erba, John H. Castellanos, Sahil Sihag, Saman Zonouz, Nils Ole Tippenhauer

**Abstract**: Unmanned Aerial Vehicles autonomously perform tasks with the use of state-of-the-art control algorithms. These control algorithms rely on the freshness and correctness of sensor readings. Incorrect control actions lead to catastrophic destabilization of the process.   In this work, we propose a multi-part \emph{Sensor Deprivation Attacks} (SDAs), aiming to stealthily impact process control via sensor reconfiguration. In the first part, the attacker will inject messages on local buses that connect to the sensor. The injected message reconfigures the sensors, e.g.,~to suspend the sensing. In the second part, those manipulation primitives are selectively used to cause adversarial sensor values at the controller, transparently to the data consumer. In the third part, the manipulated sensor values lead to unwanted control actions (e.g. a drone crash). We experimentally investigate all three parts of our proposed attack. Our findings show that i)~reconfiguring sensors can have surprising effects on reported sensor values, and ii)~the attacker can stall the overall Kalman Filter state estimation, leading to a complete stop of control computations. As a result, the UAV becomes destabilized, leading to a crash or significant deviation from its planned trajectory (over 30 meters). We also propose an attack synthesis methodology that optimizes the timing of these SDA manipulations, maximizing their impact. Notably, our results demonstrate that these SDAs evade detection by state-of-the-art UAV anomaly detectors.   Our work shows that attacks on sensors are not limited to continuously inducing random measurements, and demonstrate that sensor reconfiguration can completely stall the drone controller. In our experiments, state-of-the-art UAV controller software and countermeasures are unable to handle such manipulations. Hence, we also discuss new corresponding countermeasures.



## **15. Denial-of-Service Poisoning Attacks against Large Language Models**

cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10760v1) [paper-pdf](http://arxiv.org/pdf/2410.10760v1)

**Authors**: Kuofeng Gao, Tianyu Pang, Chao Du, Yong Yang, Shu-Tao Xia, Min Lin

**Abstract**: Recent studies have shown that LLMs are vulnerable to denial-of-service (DoS) attacks, where adversarial inputs like spelling errors or non-semantic prompts trigger endless outputs without generating an [EOS] token. These attacks can potentially cause high latency and make LLM services inaccessible to other users or tasks. However, when there are speech-to-text interfaces (e.g., voice commands to a robot), executing such DoS attacks becomes challenging, as it is difficult to introduce spelling errors or non-semantic prompts through speech. A simple DoS attack in these scenarios would be to instruct the model to "Keep repeating Hello", but we observe that relying solely on natural instructions limits output length, which is bounded by the maximum length of the LLM's supervised finetuning (SFT) data. To overcome this limitation, we propose poisoning-based DoS (P-DoS) attacks for LLMs, demonstrating that injecting a single poisoned sample designed for DoS purposes can break the output length limit. For example, a poisoned sample can successfully attack GPT-4o and GPT-4o mini (via OpenAI's finetuning API) using less than $1, causing repeated outputs up to the maximum inference length (16K tokens, compared to 0.5K before poisoning). Additionally, we perform comprehensive ablation studies on open-source LLMs and extend our method to LLM agents, where attackers can control both the finetuning dataset and algorithm. Our findings underscore the urgent need for defenses against P-DoS attacks to secure LLMs. Our code is available at https://github.com/sail-sg/P-DoS.



## **16. Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings**

cs.LG

Code and pre-trained models are available at  https://github.com/AdaptiveMotorControlLab/AROS

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10744v1) [paper-pdf](http://arxiv.org/pdf/2410.10744v1)

**Authors**: Hossein Mirzaei, Mackenzie W. Mathis

**Abstract**: Despite significant advancements in out-of-distribution (OOD) detection, existing methods still struggle to maintain robustness against adversarial attacks, compromising their reliability in critical real-world applications. Previous studies have attempted to address this challenge by exposing detectors to auxiliary OOD datasets alongside adversarial training. However, the increased data complexity inherent in adversarial training, and the myriad of ways that OOD samples can arise during testing, often prevent these approaches from establishing robust decision boundaries. To address these limitations, we propose AROS, a novel approach leveraging neural ordinary differential equations (NODEs) with Lyapunov stability theorem in order to obtain robust embeddings for OOD detection. By incorporating a tailored loss function, we apply Lyapunov stability theory to ensure that both in-distribution (ID) and OOD data converge to stable equilibrium points within the dynamical system. This approach encourages any perturbed input to return to its stable equilibrium, thereby enhancing the model's robustness against adversarial perturbations. To not use additional data, we generate fake OOD embeddings by sampling from low-likelihood regions of the ID data feature space, approximating the boundaries where OOD data are likely to reside. To then further enhance robustness, we propose the use of an orthogonal binary layer following the stable feature space, which maximizes the separation between the equilibrium points of ID and OOD samples. We validate our method through extensive experiments across several benchmarks, demonstrating superior performance, particularly under adversarial attacks. Notably, our approach improves robust detection performance from 37.8% to 80.1% on CIFAR-10 vs. CIFAR-100 and from 29.0% to 67.0% on CIFAR-100 vs. CIFAR-10.



## **17. Towards Calibrated Losses for Adversarial Robust Reject Option Classification**

cs.LG

Accepted at Asian Conference on Machine Learning (ACML) , 2024

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10736v1) [paper-pdf](http://arxiv.org/pdf/2410.10736v1)

**Authors**: Vrund Shah, Tejas Chaudhari, Naresh Manwani

**Abstract**: Robustness towards adversarial attacks is a vital property for classifiers in several applications such as autonomous driving, medical diagnosis, etc. Also, in such scenarios, where the cost of misclassification is very high, knowing when to abstain from prediction becomes crucial. A natural question is which surrogates can be used to ensure learning in scenarios where the input points are adversarially perturbed and the classifier can abstain from prediction? This paper aims to characterize and design surrogates calibrated in "Adversarial Robust Reject Option" setting. First, we propose an adversarial robust reject option loss $\ell_{d}^{\gamma}$ and analyze it for the hypothesis set of linear classifiers ($\mathcal{H}_{\textrm{lin}}$). Next, we provide a complete characterization result for any surrogate to be $(\ell_{d}^{\gamma},\mathcal{H}_{\textrm{lin}})$- calibrated. To demonstrate the difficulty in designing surrogates to $\ell_{d}^{\gamma}$, we show negative calibration results for convex surrogates and quasi-concave conditional risk cases (these gave positive calibration in adversarial setting without reject option). We also empirically argue that Shifted Double Ramp Loss (DRL) and Shifted Double Sigmoid Loss (DSL) satisfy the calibration conditions. Finally, we demonstrate the robustness of shifted DRL and shifted DSL against adversarial perturbations on a synthetically generated dataset.



## **18. Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues**

cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10700v1) [paper-pdf](http://arxiv.org/pdf/2410.10700v1)

**Authors**: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao

**Abstract**: This study exposes the safety vulnerabilities of Large Language Models (LLMs) in multi-turn interactions, where malicious users can obscure harmful intents across several queries. We introduce ActorAttack, a novel multi-turn attack method inspired by actor-network theory, which models a network of semantically linked actors as attack clues to generate diverse and effective attack paths toward harmful targets. ActorAttack addresses two main challenges in multi-turn attacks: (1) concealing harmful intents by creating an innocuous conversation topic about the actor, and (2) uncovering diverse attack paths towards the same harmful target by leveraging LLMs' knowledge to specify the correlated actors as various attack clues. In this way, ActorAttack outperforms existing single-turn and multi-turn attack methods across advanced aligned LLMs, even for GPT-o1. We will publish a dataset called SafeMTData, which includes multi-turn adversarial prompts and safety alignment data, generated by ActorAttack. We demonstrate that models safety-tuned using our safety dataset are more robust to multi-turn attacks. Code is available at https://github.com/renqibing/ActorAttack.



## **19. Enhancing Robustness in Deep Reinforcement Learning: A Lyapunov Exponent Approach**

cs.LG

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10674v1) [paper-pdf](http://arxiv.org/pdf/2410.10674v1)

**Authors**: Rory Young, Nicolas Pugeault

**Abstract**: Deep reinforcement learning agents achieve state-of-the-art performance in a wide range of simulated control tasks. However, successful applications to real-world problems remain limited. One reason for this dichotomy is because the learned policies are not robust to observation noise or adversarial attacks. In this paper, we investigate the robustness of deep RL policies to a single small state perturbation in deterministic continuous control tasks. We demonstrate that RL policies can be deterministically chaotic as small perturbations to the system state have a large impact on subsequent state and reward trajectories. This unstable non-linear behaviour has two consequences: First, inaccuracies in sensor readings, or adversarial attacks, can cause significant performance degradation; Second, even policies that show robust performance in terms of rewards may have unpredictable behaviour in practice. These two facets of chaos in RL policies drastically restrict the application of deep RL to real-world problems. To address this issue, we propose an improvement on the successful Dreamer V3 architecture, implementing a Maximal Lyapunov Exponent regularisation. This new approach reduces the chaotic state dynamics, rendering the learnt policies more resilient to sensor noise or adversarial attacks and thereby improving the suitability of Deep Reinforcement Learning for real-world applications.



## **20. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

cs.LG

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10572v1) [paper-pdf](http://arxiv.org/pdf/2410.10572v1)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.



## **21. ROSAR: An Adversarial Re-Training Framework for Robust Side-Scan Sonar Object Detection**

cs.CV

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10554v1) [paper-pdf](http://arxiv.org/pdf/2410.10554v1)

**Authors**: Martin Aubard, László Antal, Ana Madureira, Luis F. Teixeira, Erika Ábrahám

**Abstract**: This paper introduces ROSAR, a novel framework enhancing the robustness of deep learning object detection models tailored for side-scan sonar (SSS) images, generated by autonomous underwater vehicles using sonar sensors. By extending our prior work on knowledge distillation (KD), this framework integrates KD with adversarial retraining to address the dual challenges of model efficiency and robustness against SSS noises. We introduce three novel, publicly available SSS datasets, capturing different sonar setups and noise conditions. We propose and formalize two SSS safety properties and utilize them to generate adversarial datasets for retraining. Through a comparative analysis of projected gradient descent (PGD) and patch-based adversarial attacks, ROSAR demonstrates significant improvements in model robustness and detection accuracy under SSS-specific conditions, enhancing the model's robustness by up to 1.85%. ROSAR is available at https://github.com/remaro-network/ROSAR-framework.



## **22. Generalized Adversarial Code-Suggestions: Exploiting Contexts of LLM-based Code-Completion**

cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10526v1) [paper-pdf](http://arxiv.org/pdf/2410.10526v1)

**Authors**: Karl Rubel, Maximilian Noppel, Christian Wressnegger

**Abstract**: While convenient, relying on LLM-powered code assistants in day-to-day work gives rise to severe attacks. For instance, the assistant might introduce subtle flaws and suggest vulnerable code to the user. These adversarial code-suggestions can be introduced via data poisoning and, thus, unknowingly by the model creators. In this paper, we provide a generalized formulation of such attacks, spawning and extending related work in this domain. This formulation is defined over two components: First, a trigger pattern occurring in the prompts of a specific user group, and, second, a learnable map in embedding space from the prompt to an adversarial bait. The latter gives rise to novel and more flexible targeted attack-strategies, allowing the adversary to choose the most suitable trigger pattern for a specific user-group arbitrarily, without restrictions on the pattern's tokens. Our directional-map attacks and prompt-indexing attacks increase the stealthiness decisively. We extensively evaluate the effectiveness of these attacks and carefully investigate defensive mechanisms to explore the limits of generalized adversarial code-suggestions. We find that most defenses unfortunately offer little protection only.



## **23. Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks**

cs.LG

A preliminary version of this work appeared at the AdvML-Frontiers @  NeurIPS 2024 workshop

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2407.10867v2) [paper-pdf](http://arxiv.org/pdf/2407.10867v2)

**Authors**: Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan Günnemann

**Abstract**: Generalization of machine learning models can be severely compromised by data poisoning, where adversarial changes are applied to the training data. This vulnerability has led to interest in certifying (i.e., proving) that such changes up to a certain magnitude do not affect test predictions. We, for the first time, certify Graph Neural Networks (GNNs) against poisoning attacks, including backdoors, targeting the node features of a given graph. Our certificates are white-box and based upon $(i)$ the neural tangent kernel, which characterizes the training dynamics of sufficiently wide networks; and $(ii)$ a novel reformulation of the bilevel optimization problem describing poisoning as a mixed-integer linear program. Consequently, we leverage our framework to provide fundamental insights into the role of graph structure and its connectivity on the worst-case robustness behavior of convolution-based and PageRank-based GNNs. We note that our framework is more general and constitutes the first approach to derive white-box poisoning certificates for NNs, which can be of independent interest beyond graph-related tasks.



## **24. Achieving Optimal Breakdown for Byzantine Robust Gossip**

math.OC

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10418v1) [paper-pdf](http://arxiv.org/pdf/2410.10418v1)

**Authors**: Renaud Gaucher, Aymeric Dieuleveut, Hadrien Hendrikx

**Abstract**: Distributed approaches have many computational benefits, but they are vulnerable to attacks from a subset of devices transmitting incorrect information. This paper investigates Byzantine-resilient algorithms in a decentralized setting, where devices communicate directly with one another. We investigate the notion of breakdown point, and show an upper bound on the number of adversaries that decentralized algorithms can tolerate. We introduce $\mathrm{CG}^+$, an algorithm at the intersection of $\mathrm{ClippedGossip}$ and $\mathrm{NNA}$, two popular approaches for robust decentralized learning. $\mathrm{CG}^+$ meets our upper bound, and thus obtains optimal robustness guarantees, whereas neither of the existing two does. We provide experimental evidence for this gap by presenting an attack tailored to sparse graphs which breaks $\mathrm{NNA}$ but against which $\mathrm{CG}^+$ is robust.



## **25. Feature Averaging: An Implicit Bias of Gradient Descent Leading to Non-Robustness in Neural Networks**

cs.LG

78 pages, 10 figures

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10322v1) [paper-pdf](http://arxiv.org/pdf/2410.10322v1)

**Authors**: Binghui Li, Zhixuan Pan, Kaifeng Lyu, Jian Li

**Abstract**: In this work, we investigate a particular implicit bias in the gradient descent training process, which we term "Feature Averaging", and argue that it is one of the principal factors contributing to non-robustness of deep neural networks. Despite the existence of multiple discriminative features capable of classifying data, neural networks trained by gradient descent exhibit a tendency to learn the average (or certain combination) of these features, rather than distinguishing and leveraging each feature individually. In particular, we provide a detailed theoretical analysis of the training dynamics of gradient descent in a two-layer ReLU network for a binary classification task, where the data distribution consists of multiple clusters with orthogonal cluster center vectors. We rigorously prove that gradient descent converges to the regime of feature averaging, wherein the weights associated with each hidden-layer neuron represent an average of the cluster centers (each center corresponding to a distinct feature). It leads the network classifier to be non-robust due to an attack that aligns with the negative direction of the averaged features. Furthermore, we prove that, with the provision of more granular supervised information, a two-layer multi-class neural network is capable of learning individual features, from which one can derive a binary classifier with the optimal robustness under our setting. Besides, we also conduct extensive experiments using synthetic datasets, MNIST and CIFAR-10 to substantiate the phenomenon of feature averaging and its role in adversarial robustness of neural networks. We hope the theoretical and empirical insights can provide a deeper understanding of the impact of the gradient descent training on feature learning process, which in turn influences the robustness of the network, and how more detailed supervision may enhance model robustness.



## **26. DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation**

cs.CV

* denotes equal contributions; ^ denotes corresponding author. In  this updated version, we have expanded our research to include more  experiments on various adversarial attack methods and latest dataset  distillation studies. All new results have been incorporated into the  document

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2403.13322v3) [paper-pdf](http://arxiv.org/pdf/2403.13322v3)

**Authors**: Yifan Wu, Jiawei Du, Ping Liu, Yuewei Lin, Wei Xu, Wenqing Cheng

**Abstract**: Dataset distillation is an advanced technique aimed at compressing datasets into significantly smaller counterparts, while preserving formidable training performance. Significant efforts have been devoted to promote evaluation accuracy under limited compression ratio while overlooked the robustness of distilled dataset. In this work, we introduce a comprehensive benchmark that, to the best of our knowledge, is the most extensive to date for evaluating the adversarial robustness of distilled datasets in a unified way. Our benchmark significantly expands upon prior efforts by incorporating a wider range of dataset distillation methods, including the latest advancements such as TESLA and SRe2L, a diverse array of adversarial attack methods, and evaluations across a broader and more extensive collection of datasets such as ImageNet-1K. Moreover, we assessed the robustness of these distilled datasets against representative adversarial attack algorithms like PGD and AutoAttack, while exploring their resilience from a frequency perspective. We also discovered that incorporating distilled data into the training batches of the original dataset can yield to improvement of robustness.



## **27. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2405.17894v2) [paper-pdf](http://arxiv.org/pdf/2405.17894v2)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.



## **28. Out-of-Bounding-Box Triggers: A Stealthy Approach to Cheat Object Detectors**

cs.CV

ECCV 2024

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10091v1) [paper-pdf](http://arxiv.org/pdf/2410.10091v1)

**Authors**: Tao Lin, Lijia Yu, Gaojie Jin, Renjue Li, Peng Wu, Lijun Zhang

**Abstract**: In recent years, the study of adversarial robustness in object detection systems, particularly those based on deep neural networks (DNNs), has become a pivotal area of research. Traditional physical attacks targeting object detectors, such as adversarial patches and texture manipulations, directly manipulate the surface of the object. While these methods are effective, their overt manipulation of objects may draw attention in real-world applications. To address this, this paper introduces a more subtle approach: an inconspicuous adversarial trigger that operates outside the bounding boxes, rendering the object undetectable to the model. We further enhance this approach by proposing the Feature Guidance (FG) technique and the Universal Auto-PGD (UAPGD) optimization strategy for crafting high-quality triggers. The effectiveness of our method is validated through extensive empirical testing, demonstrating its high performance in both digital and physical environments. The code and video will be available at: https://github.com/linToTao/Out-of-bbox-attack.



## **29. The Role of Fake Users in Sequential Recommender Systems**

cs.IR

10 pages, 2 figures

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09936v1) [paper-pdf](http://arxiv.org/pdf/2410.09936v1)

**Authors**: Filippo Betello

**Abstract**: Sequential Recommender Systems (SRSs) are widely used to model user behavior over time, yet their robustness remains an under-explored area of research. In this paper, we conduct an empirical study to assess how the presence of fake users, who engage in random interactions, follow popular or unpopular items, or focus on a single genre, impacts the performance of SRSs in real-world scenarios. We evaluate two SRS models across multiple datasets, using established metrics such as Normalized Discounted Cumulative Gain (NDCG) and Rank Sensitivity List (RLS) to measure performance. While traditional metrics like NDCG remain relatively stable, our findings reveal that the presence of fake users severely degrades RLS metrics, often reducing them to near-zero values. These results highlight the need for further investigation into the effects of fake users on training data and emphasize the importance of developing more resilient SRSs that can withstand different types of adversarial attacks.



## **30. Extreme Miscalibration and the Illusion of Adversarial Robustness**

cs.CL

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2402.17509v3) [paper-pdf](http://arxiv.org/pdf/2402.17509v3)

**Authors**: Vyas Raina, Samson Tan, Volkan Cevher, Aditya Rawal, Sheng Zha, George Karypis

**Abstract**: Deep learning-based Natural Language Processing (NLP) models are vulnerable to adversarial attacks, where small perturbations can cause a model to misclassify. Adversarial Training (AT) is often used to increase model robustness. However, we have discovered an intriguing phenomenon: deliberately or accidentally miscalibrating models masks gradients in a way that interferes with adversarial attack search methods, giving rise to an apparent increase in robustness. We show that this observed gain in robustness is an illusion of robustness (IOR), and demonstrate how an adversary can perform various forms of test-time temperature calibration to nullify the aforementioned interference and allow the adversarial attack to find adversarial examples. Hence, we urge the NLP community to incorporate test-time temperature scaling into their robustness evaluations to ensure that any observed gains are genuine. Finally, we show how the temperature can be scaled during \textit{training} to improve genuine robustness.



## **31. Provably Reliable Conformal Prediction Sets in the Presence of Data Poisoning**

cs.LG

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09878v1) [paper-pdf](http://arxiv.org/pdf/2410.09878v1)

**Authors**: Yan Scholten, Stephan Günnemann

**Abstract**: Conformal prediction provides model-agnostic and distribution-free uncertainty quantification through prediction sets that are guaranteed to include the ground truth with any user-specified probability. Yet, conformal prediction is not reliable under poisoning attacks where adversaries manipulate both training and calibration data, which can significantly alter prediction sets in practice. As a solution, we propose reliable prediction sets (RPS): the first efficient method for constructing conformal prediction sets with provable reliability guarantees under poisoning. To ensure reliability under training poisoning, we introduce smoothed score functions that reliably aggregate predictions of classifiers trained on distinct partitions of the training data. To ensure reliability under calibration poisoning, we construct multiple prediction sets, each calibrated on distinct subsets of the calibration data. We then aggregate them into a majority prediction set, which includes a class only if it appears in a majority of the individual sets. Both proposed aggregations mitigate the influence of datapoints in the training and calibration data on the final prediction set. We experimentally validate our approach on image classification tasks, achieving strong reliability while maintaining utility and preserving coverage on clean data. Overall, our approach represents an important step towards more trustworthy uncertainty quantification in the presence of data poisoning.



## **32. Understanding Robustness of Parameter-Efficient Tuning for Image Classification**

cs.CV

5 pages, 2 figures. Work in Progress

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09845v1) [paper-pdf](http://arxiv.org/pdf/2410.09845v1)

**Authors**: Jiacheng Ruan, Xian Gao, Suncheng Xiang, Mingye Xie, Ting Liu, Yuzhuo Fu

**Abstract**: Parameter-efficient tuning (PET) techniques calibrate the model's predictions on downstream tasks by freezing the pre-trained models and introducing a small number of learnable parameters. However, despite the numerous PET methods proposed, their robustness has not been thoroughly investigated. In this paper, we systematically explore the robustness of four classical PET techniques (e.g., VPT, Adapter, AdaptFormer, and LoRA) under both white-box attacks and information perturbations. For white-box attack scenarios, we first analyze the performance of PET techniques using FGSM and PGD attacks. Subsequently, we further explore the transferability of adversarial samples and the impact of learnable parameter quantities on the robustness of PET methods. Under information perturbation attacks, we introduce four distinct perturbation strategies, including Patch-wise Drop, Pixel-wise Drop, Patch Shuffle, and Gaussian Noise, to comprehensively assess the robustness of these PET techniques in the presence of information loss. Via these extensive studies, we enhance the understanding of the robustness of PET methods, providing valuable insights for improving their performance in computer vision applications. The code is available at https://github.com/JCruan519/PETRobustness.



## **33. Robust 3D Point Clouds Classification based on Declarative Defenders**

cs.CV

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09691v1) [paper-pdf](http://arxiv.org/pdf/2410.09691v1)

**Authors**: Kaidong Li, Tianxiao Zhang, Chuncong Zhong, Ziming Zhang, Guanghui Wang

**Abstract**: 3D point cloud classification requires distinct models from 2D image classification due to the divergent characteristics of the respective input data. While 3D point clouds are unstructured and sparse, 2D images are structured and dense. Bridging the domain gap between these two data types is a non-trivial challenge to enable model interchangeability. Recent research using Lattice Point Classifier (LPC) highlights the feasibility of cross-domain applicability. However, the lattice projection operation in LPC generates 2D images with disconnected projected pixels. In this paper, we explore three distinct algorithms for mapping 3D point clouds into 2D images. Through extensive experiments, we thoroughly examine and analyze their performance and defense mechanisms. Leveraging current large foundation models, we scrutinize the feature disparities between regular 2D images and projected 2D images. The proposed approaches demonstrate superior accuracy and robustness against adversarial attacks. The generative model-based mapping algorithms yield regular 2D images, further minimizing the domain gap from regular 2D classification tasks. The source code is available at https://github.com/KaidongLi/pytorch-LatticePointClassifier.git.



## **34. Uncovering Attacks and Defenses in Secure Aggregation for Federated Deep Learning**

cs.CR

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09676v1) [paper-pdf](http://arxiv.org/pdf/2410.09676v1)

**Authors**: Yiwei Zhang, Rouzbeh Behnia, Attila A. Yavuz, Reza Ebrahimi, Elisa Bertino

**Abstract**: Federated learning enables the collaborative learning of a global model on diverse data, preserving data locality and eliminating the need to transfer user data to a central server. However, data privacy remains vulnerable, as attacks can target user training data by exploiting the updates sent by users during each learning iteration. Secure aggregation protocols are designed to mask/encrypt user updates and enable a central server to aggregate the masked information. MicroSecAgg (PoPETS 2024) proposes a single server secure aggregation protocol that aims to mitigate the high communication complexity of the existing approaches by enabling a one-time setup of the secret to be re-used in multiple training iterations. In this paper, we identify a security flaw in the MicroSecAgg that undermines its privacy guarantees. We detail the security flaw and our attack, demonstrating how an adversary can exploit predictable masking values to compromise user privacy. Our findings highlight the critical need for enhanced security measures in secure aggregation protocols, particularly the implementation of dynamic and unpredictable masking strategies. We propose potential countermeasures to mitigate these vulnerabilities and ensure robust privacy protection in the secure aggregation frameworks.



## **35. Unlearn and Burn: Adversarial Machine Unlearning Requests Destroy Model Accuracy**

cs.CR

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2410.09591v1) [paper-pdf](http://arxiv.org/pdf/2410.09591v1)

**Authors**: Yangsibo Huang, Daogao Liu, Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Milad Nasr, Amer Sinha, Chiyuan Zhang

**Abstract**: Machine unlearning algorithms, designed for selective removal of training data from models, have emerged as a promising approach to growing privacy concerns. In this work, we expose a critical yet underexplored vulnerability in the deployment of unlearning systems: the assumption that the data requested for removal is always part of the original training set. We present a threat model where an attacker can degrade model accuracy by submitting adversarial unlearning requests for data not present in the training set. We propose white-box and black-box attack algorithms and evaluate them through a case study on image classification tasks using the CIFAR-10 and ImageNet datasets, targeting a family of widely used unlearning methods. Our results show extremely poor test accuracy following the attack: 3.6% on CIFAR-10 and 0.4% on ImageNet for white-box attacks, and 8.5% on CIFAR-10 and 1.3% on ImageNet for black-box attacks. Additionally, we evaluate various verification mechanisms to detect the legitimacy of unlearning requests and reveal the challenges in verification, as most of the mechanisms fail to detect stealthy attacks without severely impairing their ability to process valid requests. These findings underscore the urgent need for research on more robust request verification methods and unlearning protocols, should the deployment of machine unlearning systems become more prevalent in the future.



## **36. Differentially Private and Byzantine-Resilient Decentralized Nonconvex Optimization: System Modeling, Utility, Resilience, and Privacy Analysis**

math.OC

13 pages, 13 figures

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2409.18632v5) [paper-pdf](http://arxiv.org/pdf/2409.18632v5)

**Authors**: Jinhui Hu, Guo Chen, Huaqing Li, Huqiang Cheng, Xiaoyu Guo, Tingwen Huang

**Abstract**: Privacy leakage and Byzantine failures are two adverse factors to the intelligent decision-making process of multi-agent systems (MASs). Considering the presence of these two issues, this paper targets the resolution of a class of nonconvex optimization problems under the Polyak-{\L}ojasiewicz (P-{\L}) condition. To address this problem, we first identify and construct the adversary system model. To enhance the robustness of stochastic gradient descent methods, we mask the local gradients with Gaussian noises and adopt a resilient aggregation method self-centered clipping (SCC) to design a differentially private (DP) decentralized Byzantine-resilient algorithm, namely DP-SCC-PL, which simultaneously achieves differential privacy and Byzantine resilience. The convergence analysis of DP-SCC-PL is challenging since the convergence error can be contributed jointly by privacy-preserving and Byzantine-resilient mechanisms, as well as the nonconvex relaxation, which is addressed via seeking the contraction relationships among the disagreement measure of reliable agents before and after aggregation, together with the optimal gap. Theoretical results reveal that DP-SCC-PL achieves consensus among all reliable agents and sublinear (inexact) convergence with well-designed step-sizes. It has also been proved that if there are no privacy issues and Byzantine agents, then the asymptotic exact convergence can be recovered. Numerical experiments verify the utility, resilience, and differential privacy of DP-SCC-PL by tackling a nonconvex optimization problem satisfying the P-{\L} condition under various Byzantine attacks.



## **37. Minimax rates of convergence for nonparametric regression under adversarial attacks**

math.ST

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2410.09402v1) [paper-pdf](http://arxiv.org/pdf/2410.09402v1)

**Authors**: Jingfu Peng, Yuhong Yang

**Abstract**: Recent research shows the susceptibility of machine learning models to adversarial attacks, wherein minor but maliciously chosen perturbations of the input can significantly degrade model performance. In this paper, we theoretically analyse the limits of robustness against such adversarial attacks in a nonparametric regression setting, by examining the minimax rates of convergence in an adversarial sup-norm. Our work reveals that the minimax rate under adversarial attacks in the input is the same as sum of two terms: one represents the minimax rate in the standard setting without adversarial attacks, and the other reflects the maximum deviation of the true regression function value within the target function class when subjected to the input perturbations. The optimal rates under the adversarial setup can be achieved by a plug-in procedure constructed from a minimax optimal estimator in the corresponding standard setting. Two specific examples are given to illustrate the established minimax results.



## **38. Targeted Attack Improves Protection against Unauthorized Diffusion Customization**

cs.CV

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2310.04687v4) [paper-pdf](http://arxiv.org/pdf/2310.04687v4)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu

**Abstract**: Diffusion models build a new milestone for image generation yet raising public concerns, for they can be fine-tuned on unauthorized images for customization. Protection based on adversarial attacks rises to encounter this unauthorized diffusion customization, by adding protective watermarks to images and poisoning diffusion models. However, current protection, leveraging untargeted attacks, does not appear to be effective enough. In this paper, we propose a simple yet effective improvement for the protection against unauthorized diffusion customization by introducing targeted attacks. We show that by carefully selecting the target, targeted attacks significantly outperform untargeted attacks in poisoning diffusion models and degrading the customization image quality. Extensive experiments validate the superiority of our method on two mainstream customization methods of diffusion models, compared to existing protections. To explain the surprising success of targeted attacks, we delve into the mechanism of attack-based protections and propose a hypothesis based on our observation, which enhances the comprehension of attack-based protections. To the best of our knowledge, we are the first to both reveal the vulnerability of diffusion models to targeted attacks and leverage targeted attacks to enhance protection against unauthorized diffusion customization. Our code is available on GitHub: \url{https://github.com/psyker-team/mist-v2}.



## **39. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

cs.SD

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.04482v2) [paper-pdf](http://arxiv.org/pdf/2407.04482v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.



## **40. On the Adversarial Transferability of Generalized "Skip Connections"**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08950v1) [paper-pdf](http://arxiv.org/pdf/2410.08950v1)

**Authors**: Yisen Wang, Yichuan Mo, Dongxian Wu, Mingjie Li, Xingjun Ma, Zhouchen Lin

**Abstract**: Skip connection is an essential ingredient for modern deep models to be deeper and more powerful. Despite their huge success in normal scenarios (state-of-the-art classification performance on natural examples), we investigate and identify an interesting property of skip connections under adversarial scenarios, namely, the use of skip connections allows easier generation of highly transferable adversarial examples. Specifically, in ResNet-like models (with skip connections), we find that using more gradients from the skip connections rather than the residual modules according to a decay factor during backpropagation allows one to craft adversarial examples with high transferability. The above method is termed as Skip Gradient Method (SGM). Although starting from ResNet-like models in vision domains, we further extend SGM to more advanced architectures, including Vision Transformers (ViTs) and models with length-varying paths and other domains, i.e. natural language processing. We conduct comprehensive transfer attacks against various models including ResNets, Transformers, Inceptions, Neural Architecture Search, and Large Language Models (LLMs). We show that employing SGM can greatly improve the transferability of crafted attacks in almost all cases. Furthermore, considering the big complexity for practical use, we further demonstrate that SGM can even improve the transferability on ensembles of models or targeted attacks and the stealthiness against current defenses. At last, we provide theoretical explanations and empirical insights on how SGM works. Our findings not only motivate new adversarial research into the architectural characteristics of models but also open up further challenges for secure model architecture design. Our code is available at https://github.com/mo666666/SGM.



## **41. Fragile Giants: Understanding the Susceptibility of Models to Subpopulation Attacks**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08872v1) [paper-pdf](http://arxiv.org/pdf/2410.08872v1)

**Authors**: Isha Gupta, Hidde Lycklama, Emanuel Opel, Evan Rose, Anwar Hithnawi

**Abstract**: As machine learning models become increasingly complex, concerns about their robustness and trustworthiness have become more pressing. A critical vulnerability of these models is data poisoning attacks, where adversaries deliberately alter training data to degrade model performance. One particularly stealthy form of these attacks is subpopulation poisoning, which targets distinct subgroups within a dataset while leaving overall performance largely intact. The ability of these attacks to generalize within subpopulations poses a significant risk in real-world settings, as they can be exploited to harm marginalized or underrepresented groups within the dataset. In this work, we investigate how model complexity influences susceptibility to subpopulation poisoning attacks. We introduce a theoretical framework that explains how overparameterized models, due to their large capacity, can inadvertently memorize and misclassify targeted subpopulations. To validate our theory, we conduct extensive experiments on large-scale image and text datasets using popular model architectures. Our results show a clear trend: models with more parameters are significantly more vulnerable to subpopulation poisoning. Moreover, we find that attacks on smaller, human-interpretable subgroups often go undetected by these models. These results highlight the need to develop defenses that specifically address subpopulation vulnerabilities.



## **42. The Good, the Bad and the Ugly: Watermarks, Transferable Attacks and Adversarial Defenses**

cs.LG

42 pages, 6 figures, preliminary version published in ICML 2024  (Workshop on Theoretical Foundations of Foundation Models), see  https://openreview.net/pdf?id=WMaFRiggwV

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08864v1) [paper-pdf](http://arxiv.org/pdf/2410.08864v1)

**Authors**: Grzegorz Głuch, Berkant Turan, Sai Ganesh Nagarajan, Sebastian Pokutta

**Abstract**: We formalize and extend existing definitions of backdoor-based watermarks and adversarial defenses as interactive protocols between two players. The existence of these schemes is inherently tied to the learning tasks for which they are designed. Our main result shows that for almost every discriminative learning task, at least one of the two -- a watermark or an adversarial defense -- exists. The term "almost every" indicates that we also identify a third, counterintuitive but necessary option, i.e., a scheme we call a transferable attack. By transferable attack, we refer to an efficient algorithm computing queries that look indistinguishable from the data distribution and fool all efficient defenders. To this end, we prove the necessity of a transferable attack via a construction that uses a cryptographic tool called homomorphic encryption. Furthermore, we show that any task that satisfies our notion of a transferable attack implies a cryptographic primitive, thus requiring the underlying task to be computationally complex. These two facts imply an "equivalence" between the existence of transferable attacks and cryptography. Finally, we show that the class of tasks of bounded VC-dimension has an adversarial defense, and a subclass of them has a watermark.



## **43. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08827v1) [paper-pdf](http://arxiv.org/pdf/2410.08827v1)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.



## **44. Training on Fake Labels: Mitigating Label Leakage in Split Learning via Secure Dimension Transformation**

cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.09125v1) [paper-pdf](http://arxiv.org/pdf/2410.09125v1)

**Authors**: Yukun Jiang, Peiran Wang, Chengguo Lin, Ziyue Huang, Yong Cheng

**Abstract**: Two-party split learning has emerged as a popular paradigm for vertical federated learning. To preserve the privacy of the label owner, split learning utilizes a split model, which only requires the exchange of intermediate representations (IRs) based on the inputs and gradients for each IR between two parties during the learning process. However, split learning has recently been proven to survive label inference attacks. Though several defense methods could be adopted, they either have limited defensive performance or significantly negatively impact the original mission. In this paper, we propose a novel two-party split learning method to defend against existing label inference attacks while maintaining the high utility of the learned models. Specifically, we first craft a dimension transformation module, SecDT, which could achieve bidirectional mapping between original labels and increased K-class labels to mitigate label leakage from the directional perspective. Then, a gradient normalization algorithm is designed to remove the magnitude divergence of gradients from different classes. We propose a softmax-normalized Gaussian noise to mitigate privacy leakage and make our K unknowable to adversaries. We conducted experiments on real-world datasets, including two binary-classification datasets (Avazu and Criteo) and three multi-classification datasets (MNIST, FashionMNIST, CIFAR-10); we also considered current attack schemes, including direction, norm, spectral, and model completion attacks. The detailed experiments demonstrate our proposed method's effectiveness and superiority over existing approaches. For instance, on the Avazu dataset, the attack AUC of evaluated four prominent attacks could be reduced by 0.4532+-0.0127.



## **45. Natural Language Induced Adversarial Images**

cs.CR

Carmera-ready version. To appear in ACM MM 2024

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08620v1) [paper-pdf](http://arxiv.org/pdf/2410.08620v1)

**Authors**: Xiaopei Zhu, Peiyang Xu, Guanning Zeng, Yingpeng Dong, Xiaolin Hu

**Abstract**: Research of adversarial attacks is important for AI security because it shows the vulnerability of deep learning models and helps to build more robust models. Adversarial attacks on images are most widely studied, which include noise-based attacks, image editing-based attacks, and latent space-based attacks. However, the adversarial examples crafted by these methods often lack sufficient semantic information, making it challenging for humans to understand the failure modes of deep learning models under natural conditions. To address this limitation, we propose a natural language induced adversarial image attack method. The core idea is to leverage a text-to-image model to generate adversarial images given input prompts, which are maliciously constructed to lead to misclassification for a target model. To adopt commercial text-to-image models for synthesizing more natural adversarial images, we propose an adaptive genetic algorithm (GA) for optimizing discrete adversarial prompts without requiring gradients and an adaptive word space reduction method for improving query efficiency. We further used CLIP to maintain the semantic consistency of the generated images. In our experiments, we found that some high-frequency semantic information such as "foggy", "humid", "stretching", etc. can easily cause classifier errors. This adversarial semantic information exists not only in generated images but also in photos captured in the real world. We also found that some adversarial semantic information can be transferred to unknown classification tasks. Furthermore, our attack method can transfer to different text-to-image models (e.g., Midjourney, DALL-E 3, etc.) and image classifiers. Our code is available at: https://github.com/zxp555/Natural-Language-Induced-Adversarial-Images.



## **46. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2405.19360v3) [paper-pdf](http://arxiv.org/pdf/2405.19360v3)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.



## **47. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

cs.CL

12 pages, 9 figures

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.21659v3) [paper-pdf](http://arxiv.org/pdf/2407.21659v3)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.



## **48. NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic**

cs.CL

Published as a conference paper at ACL 2023

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2307.02849v2) [paper-pdf](http://arxiv.org/pdf/2307.02849v2)

**Authors**: Zi'ou Zheng, Xiaodan Zhu

**Abstract**: Reasoning has been a central topic in artificial intelligence from the beginning. The recent progress made on distributed representation and neural networks continues to improve the state-of-the-art performance of natural language inference. However, it remains an open question whether the models perform real reasoning to reach their conclusions or rely on spurious correlations. Adversarial attacks have proven to be an important tool to help evaluate the Achilles' heel of the victim models. In this study, we explore the fundamental problem of developing attack models based on logic formalism. We propose NatLogAttack to perform systematic attacks centring around natural logic, a classical logic formalism that is traceable back to Aristotle's syllogism and has been closely developed for natural language inference. The proposed framework renders both label-preserving and label-flipping attacks. We show that compared to the existing attack models, NatLogAttack generates better adversarial examples with fewer visits to the victim models. The victim models are found to be more vulnerable under the label-flipping setting. NatLogAttack provides a tool to probe the existing and future NLI models' capacity from a key viewpoint and we hope more logic-based attacks will be further explored for understanding the desired property of reasoning.



## **49. Backdooring Bias into Text-to-Image Models**

cs.LG

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2406.15213v2) [paper-pdf](http://arxiv.org/pdf/2406.15213v2)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasaryan, Amir Houmansadr

**Abstract**: Text-conditional diffusion models, i.e. text-to-image, produce eye-catching images that represent descriptions given by a user. These images often depict benign concepts but could also carry other purposes. Specifically, visual information is easy to comprehend and could be weaponized for propaganda -- a serious challenge given widespread usage and deployment of generative models. In this paper, we show that an adversary can add an arbitrary bias through a backdoor attack that would affect even benign users generating images. While a user could inspect a generated image to comply with the given text description, our attack remains stealthy as it preserves semantic information given in the text prompt. Instead, a compromised model modifies other unspecified features of the image to add desired biases (that increase by 4-8x). Furthermore, we show how the current state-of-the-art generative models make this attack both cheap and feasible for any adversary, with costs ranging between $12-$18. We evaluate our attack over various types of triggers, adversary objectives, and biases and discuss mitigations and future work. Our code is available at https://github.com/jrohsc/Backdororing_Bias.



## **50. Time Traveling to Defend Against Adversarial Example Attacks in Image Classification**

cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.08338v1) [paper-pdf](http://arxiv.org/pdf/2410.08338v1)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial example attacks have emerged as a critical threat to machine learning. Adversarial attacks in image classification abuse various, minor modifications to the image that confuse the image classification neural network -- while the image still remains recognizable to humans. One important domain where the attacks have been applied is in the automotive setting with traffic sign classification. Researchers have demonstrated that adding stickers, shining light, or adding shadows are all different means to make machine learning inference algorithms mis-classify the traffic signs. This can cause potentially dangerous situations as a stop sign is recognized as a speed limit sign causing vehicles to ignore it and potentially leading to accidents. To address these attacks, this work focuses on enhancing defenses against such adversarial attacks. This work shifts the advantage to the user by introducing the idea of leveraging historical images and majority voting. While the attacker modifies a traffic sign that is currently being processed by the victim's machine learning inference, the victim can gain advantage by examining past images of the same traffic sign. This work introduces the notion of ''time traveling'' and uses historical Street View images accessible to anybody to perform inference on different, past versions of the same traffic sign. In the evaluation, the proposed defense has 100% effectiveness against latest adversarial example attack on traffic sign classification algorithm.



