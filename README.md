# Latest Adversarial Attack Papers
**update at 2024-10-21 09:53:25**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Distributionally and Adversarially Robust Logistic Regression via Intersecting Wasserstein Balls**

math.OC

33 pages, 3 color figures, under review at a conference

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2407.13625v2) [paper-pdf](http://arxiv.org/pdf/2407.13625v2)

**Authors**: Aras Selvi, Eleonora Kreacic, Mohsen Ghassemi, Vamsi Potluru, Tucker Balch, Manuela Veloso

**Abstract**: Adversarially robust optimization (ARO) has become the de facto standard for training models to defend against adversarial attacks during testing. However, despite their robustness, these models often suffer from severe overfitting. To mitigate this issue, several successful approaches have been proposed, including replacing the empirical distribution in training with: (i) a worst-case distribution within an ambiguity set, leading to a distributionally robust (DR) counterpart of ARO; or (ii) a mixture of the empirical distribution with one derived from an auxiliary dataset (e.g., synthetic, external, or out-of-domain). Building on the first approach, we explore the Wasserstein DR counterpart of ARO for logistic regression and show it admits a tractable convex optimization reformulation. Adopting the second approach, we enhance the DR framework by intersecting its ambiguity set with one constructed from an auxiliary dataset, which yields significant improvements when the Wasserstein distance between the data-generating and auxiliary distributions can be estimated. We analyze the resulting optimization problem, develop efficient solutions, and show that our method outperforms benchmark approaches on standard datasets.



## **2. Explainable Graph Neural Networks Under Fire**

cs.LG

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2406.06417v2) [paper-pdf](http://arxiv.org/pdf/2406.06417v2)

**Authors**: Zhong Li, Simon Geisler, Yuhang Wang, Stephan Günnemann, Matthijs van Leeuwen

**Abstract**: Predictions made by graph neural networks (GNNs) usually lack interpretability due to their complex computational behavior and the abstract nature of graphs. In an attempt to tackle this, many GNN explanation methods have emerged. Their goal is to explain a model's predictions and thereby obtain trust when GNN models are deployed in decision critical applications. Most GNN explanation methods work in a post-hoc manner and provide explanations in the form of a small subset of important edges and/or nodes. In this paper we demonstrate that these explanations can unfortunately not be trusted, as common GNN explanation methods turn out to be highly susceptible to adversarial perturbations. That is, even small perturbations of the original graph structure that preserve the model's predictions may yield drastically different explanations. This calls into question the trustworthiness and practical utility of post-hoc explanation methods for GNNs. To be able to attack GNN explanation models, we devise a novel attack method dubbed \textit{GXAttack}, the first \textit{optimization-based} adversarial white-box attack method for post-hoc GNN explanations under such settings. Due to the devastating effectiveness of our attack, we call for an adversarial evaluation of future GNN explainers to demonstrate their robustness. For reproducibility, our code is available via GitHub.



## **3. JAILJUDGE: A Comprehensive Jailbreak Judge Benchmark with Multi-Agent Enhanced Explanation Evaluation Framework**

cs.CL

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.12855v2) [paper-pdf](http://arxiv.org/pdf/2410.12855v2)

**Authors**: Fan Liu, Yue Feng, Zhao Xu, Lixin Su, Xinyu Ma, Dawei Yin, Hao Liu

**Abstract**: Despite advancements in enhancing LLM safety against jailbreak attacks, evaluating LLM defenses remains a challenge, with current methods often lacking explainability and generalization to complex scenarios, leading to incomplete assessments (e.g., direct judgment without reasoning, low F1 score of GPT-4 in complex cases, bias in multilingual scenarios). To address this, we present JAILJUDGE, a comprehensive benchmark featuring diverse risk scenarios, including synthetic, adversarial, in-the-wild, and multilingual prompts, along with high-quality human-annotated datasets. The JAILJUDGE dataset includes over 35k+ instruction-tune data with reasoning explainability and JAILJUDGETEST, a 4.5k+ labeled set for risk scenarios, and a 6k+ multilingual set across ten languages. To enhance evaluation with explicit reasoning, we propose the JailJudge MultiAgent framework, which enables explainable, fine-grained scoring (1 to 10). This framework supports the construction of instruction-tuning ground truth and facilitates the development of JAILJUDGE Guard, an end-to-end judge model that provides reasoning and eliminates API costs. Additionally, we introduce JailBoost, an attacker-agnostic attack enhancer, and GuardShield, a moderation defense, both leveraging JAILJUDGE Guard. Our experiments demonstrate the state-of-the-art performance of JailJudge methods (JailJudge MultiAgent, JAILJUDGE Guard) across diverse models (e.g., GPT-4, Llama-Guard) and zero-shot scenarios. JailBoost and GuardShield significantly improve jailbreak attack and defense tasks under zero-shot settings, with JailBoost enhancing performance by 29.24% and GuardShield reducing defense ASR from 40.46% to 0.15%.



## **4. DMGNN: Detecting and Mitigating Backdoor Attacks in Graph Neural Networks**

cs.CR

12 pages, 8 figures

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14105v1) [paper-pdf](http://arxiv.org/pdf/2410.14105v1)

**Authors**: Hao Sui, Bing Chen, Jiale Zhang, Chengcheng Zhu, Di Wu, Qinghua Lu, Guodong Long

**Abstract**: Recent studies have revealed that GNNs are highly susceptible to multiple adversarial attacks. Among these, graph backdoor attacks pose one of the most prominent threats, where attackers cause models to misclassify by learning the backdoored features with injected triggers and modified target labels during the training phase. Based on the features of the triggers, these attacks can be categorized into out-of-distribution (OOD) and in-distribution (ID) graph backdoor attacks, triggers with notable differences from the clean sample feature distributions constitute OOD backdoor attacks, whereas the triggers in ID backdoor attacks are nearly identical to the clean sample feature distributions. Existing methods can successfully defend against OOD backdoor attacks by comparing the feature distribution of triggers and clean samples but fail to mitigate stealthy ID backdoor attacks. Due to the lack of proper supervision signals, the main task accuracy is negatively affected in defending against ID backdoor attacks. To bridge this gap, we propose DMGNN against OOD and ID graph backdoor attacks that can powerfully eliminate stealthiness to guarantee defense effectiveness and improve the model performance. Specifically, DMGNN can easily identify the hidden ID and OOD triggers via predicting label transitions based on counterfactual explanation. To further filter the diversity of generated explainable graphs and erase the influence of the trigger features, we present a reverse sampling pruning method to screen and discard the triggers directly on the data level. Extensive experimental evaluations on open graph datasets demonstrate that DMGNN far outperforms the state-of-the-art (SOTA) defense methods, reducing the attack success rate to 5% with almost negligible degradation in model performance (within 3.5%).



## **5. MMAD-Purify: A Precision-Optimized Framework for Efficient and Scalable Multi-Modal Attacks**

cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.14089v1) [paper-pdf](http://arxiv.org/pdf/2410.14089v1)

**Authors**: Xinxin Liu, Zhongliang Guo, Siyuan Huang, Chun Pong Lau

**Abstract**: Neural networks have achieved remarkable performance across a wide range of tasks, yet they remain susceptible to adversarial perturbations, which pose significant risks in safety-critical applications. With the rise of multimodality, diffusion models have emerged as powerful tools not only for generative tasks but also for various applications such as image editing, inpainting, and super-resolution. However, these models still lack robustness due to limited research on attacking them to enhance their resilience. Traditional attack techniques, such as gradient-based adversarial attacks and diffusion model-based methods, are hindered by computational inefficiencies and scalability issues due to their iterative nature. To address these challenges, we introduce an innovative framework that leverages the distilled backbone of diffusion models and incorporates a precision-optimized noise predictor to enhance the effectiveness of our attack framework. This approach not only enhances the attack's potency but also significantly reduces computational costs. Our framework provides a cutting-edge solution for multi-modal adversarial attacks, ensuring reduced latency and the generation of high-fidelity adversarial examples with superior success rates. Furthermore, we demonstrate that our framework achieves outstanding transferability and robustness against purification defenses, outperforming existing gradient-based attack models in both effectiveness and efficiency.



## **6. Uncovering Attacks and Defenses in Secure Aggregation for Federated Deep Learning**

cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.09676v2) [paper-pdf](http://arxiv.org/pdf/2410.09676v2)

**Authors**: Yiwei Zhang, Rouzbeh Behnia, Attila A. Yavuz, Reza Ebrahimi, Elisa Bertino

**Abstract**: Federated learning enables the collaborative learning of a global model on diverse data, preserving data locality and eliminating the need to transfer user data to a central server. However, data privacy remains vulnerable, as attacks can target user training data by exploiting the updates sent by users during each learning iteration. Secure aggregation protocols are designed to mask/encrypt user updates and enable a central server to aggregate the masked information. MicroSecAgg (PoPETS 2024) proposes a single server secure aggregation protocol that aims to mitigate the high communication complexity of the existing approaches by enabling a one-time setup of the secret to be re-used in multiple training iterations. In this paper, we identify a security flaw in the MicroSecAgg that undermines its privacy guarantees. We detail the security flaw and our attack, demonstrating how an adversary can exploit predictable masking values to compromise user privacy. Our findings highlight the critical need for enhanced security measures in secure aggregation protocols, particularly the implementation of dynamic and unpredictable masking strategies. We propose potential countermeasures to mitigate these vulnerabilities and ensure robust privacy protection in the secure aggregation frameworks.



## **7. Adversarial Inception for Bounded Backdoor Poisoning in Deep Reinforcement Learning**

cs.LG

10 pages, 5 figures, ICLR 2025

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13995v1) [paper-pdf](http://arxiv.org/pdf/2410.13995v1)

**Authors**: Ethan Rathbun, Christopher Amato, Alina Oprea

**Abstract**: Recent works have demonstrated the vulnerability of Deep Reinforcement Learning (DRL) algorithms against training-time, backdoor poisoning attacks. These attacks induce pre-determined, adversarial behavior in the agent upon observing a fixed trigger during deployment while allowing the agent to solve its intended task during training. Prior attacks rely on arbitrarily large perturbations to the agent's rewards to achieve both of these objectives - leaving them open to detection. Thus, in this work, we propose a new class of backdoor attacks against DRL which achieve state of the art performance while minimally altering the agent's rewards. These ``inception'' attacks train the agent to associate the targeted adversarial behavior with high returns by inducing a disjunction between the agent's chosen action and the true action executed in the environment during training. We formally define these attacks and prove they can achieve both adversarial objectives. We then devise an online inception attack which significantly out-performs prior attacks under bounded reward constraints.



## **8. Trojan Prompt Attacks on Graph Neural Networks**

cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13974v1) [paper-pdf](http://arxiv.org/pdf/2410.13974v1)

**Authors**: Minhua Lin, Zhiwei Zhang, Enyan Dai, Zongyu Wu, Yilong Wang, Xiang Zhang, Suhang Wang

**Abstract**: Graph Prompt Learning (GPL) has been introduced as a promising approach that uses prompts to adapt pre-trained GNN models to specific downstream tasks without requiring fine-tuning of the entire model. Despite the advantages of GPL, little attention has been given to its vulnerability to backdoor attacks, where an adversary can manipulate the model's behavior by embedding hidden triggers. Existing graph backdoor attacks rely on modifying model parameters during training, but this approach is impractical in GPL as GNN encoder parameters are frozen after pre-training. Moreover, downstream users may fine-tune their own task models on clean datasets, further complicating the attack. In this paper, we propose TGPA, a backdoor attack framework designed specifically for GPL. TGPA injects backdoors into graph prompts without modifying pre-trained GNN encoders and ensures high attack success rates and clean accuracy. To address the challenge of model fine-tuning by users, we introduce a finetuning-resistant poisoning approach that maintains the effectiveness of the backdoor even after downstream model adjustments. Extensive experiments on multiple datasets under various settings demonstrate the effectiveness of TGPA in compromising GPL models with fixed GNN encoders.



## **9. Multi-style conversion for semantic segmentation of lesions in fundus images by adversarial attacks**

cs.CV

preprint

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13822v1) [paper-pdf](http://arxiv.org/pdf/2410.13822v1)

**Authors**: Clément Playout, Renaud Duval, Marie Carole Boucher, Farida Cheriet

**Abstract**: The diagnosis of diabetic retinopathy, which relies on fundus images, faces challenges in achieving transparency and interpretability when using a global classification approach. However, segmentation-based databases are significantly more expensive to acquire and combining them is often problematic. This paper introduces a novel method, termed adversarial style conversion, to address the lack of standardization in annotation styles across diverse databases. By training a single architecture on combined databases, the model spontaneously modifies its segmentation style depending on the input, demonstrating the ability to convert among different labeling styles. The proposed methodology adds a linear probe to detect dataset origin based on encoder features and employs adversarial attacks to condition the model's segmentation style. Results indicate significant qualitative and quantitative through dataset combination, offering avenues for improved model generalization, uncertainty estimation and continuous interpolation between annotation styles. Our approach enables training a segmentation model with diverse databases while controlling and leveraging annotation styles for improved retinopathy diagnosis.



## **10. Persistent Pre-Training Poisoning of LLMs**

cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13722v1) [paper-pdf](http://arxiv.org/pdf/2410.13722v1)

**Authors**: Yiming Zhang, Javier Rando, Ivan Evtimov, Jianfeng Chi, Eric Michael Smith, Nicholas Carlini, Florian Tramèr, Daphne Ippolito

**Abstract**: Large language models are pre-trained on uncurated text datasets consisting of trillions of tokens scraped from the Web. Prior work has shown that: (1) web-scraped pre-training datasets can be practically poisoned by malicious actors; and (2) adversaries can compromise language models after poisoning fine-tuning datasets. Our work evaluates for the first time whether language models can also be compromised during pre-training, with a focus on the persistence of pre-training attacks after models are fine-tuned as helpful and harmless chatbots (i.e., after SFT and DPO). We pre-train a series of LLMs from scratch to measure the impact of a potential poisoning adversary under four different attack objectives (denial-of-service, belief manipulation, jailbreaking, and prompt stealing), and across a wide range of model sizes (from 600M to 7B). Our main result is that poisoning only 0.1% of a model's pre-training dataset is sufficient for three out of four attacks to measurably persist through post-training. Moreover, simple attacks like denial-of-service persist through post-training with a poisoning rate of only 0.001%.



## **11. Optimal MEV Extraction Using Absolute Commitments**

cs.GT

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13624v1) [paper-pdf](http://arxiv.org/pdf/2410.13624v1)

**Authors**: Daji Landis, Nikolaj I. Schwartzbach

**Abstract**: We propose a new, more potent attack on decentralized exchanges. This attack leverages absolute commitments, which are commitments that can condition on the strategies made by other agents. This attack allows an adversary to charge monopoly prices by committing to undercut those other miners that refuse to charge an even higher fee. This allows the miner to extract the maximum possible price from the user, potentially through side channels that evade the inefficiencies and fees usually incurred. This is considerably more efficient than the prevailing strategy of `sandwich attacks', wherein the adversary induces and profits from fluctuations in the market price to the detriment of users. The attack we propose can, in principle, be realized by the irrevocable and self-executing nature of smart contracts, which are readily available on many major blockchains. Thus, the attack could potentially be used against a decentralized exchange and could drastically reduce the utility of the affected exchange.



## **12. Transformer-Based Approaches for Sensor-Based Human Activity Recognition: Opportunities and Challenges**

cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13605v1) [paper-pdf](http://arxiv.org/pdf/2410.13605v1)

**Authors**: Clayton Souza Leite, Henry Mauranen, Aziza Zhanabatyrova, Yu Xiao

**Abstract**: Transformers have excelled in natural language processing and computer vision, paving their way to sensor-based Human Activity Recognition (HAR). Previous studies show that transformers outperform their counterparts exclusively when they harness abundant data or employ compute-intensive optimization algorithms. However, neither of these scenarios is viable in sensor-based HAR due to the scarcity of data in this field and the frequent need to perform training and inference on resource-constrained devices. Our extensive investigation into various implementations of transformer-based versus non-transformer-based HAR using wearable sensors, encompassing more than 500 experiments, corroborates these concerns. We observe that transformer-based solutions pose higher computational demands, consistently yield inferior performance, and experience significant performance degradation when quantized to accommodate resource-constrained devices. Additionally, transformers demonstrate lower robustness to adversarial attacks, posing a potential threat to user trust in HAR.



## **13. Adversarial Exposure Attack on Diabetic Retinopathy Imagery Grading**

cs.CV

13 pages, 7 figures

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2009.09231v2) [paper-pdf](http://arxiv.org/pdf/2009.09231v2)

**Authors**: Yupeng Cheng, Qing Guo, Felix Juefei-Xu, Huazhu Fu, Shang-Wei Lin, Weisi Lin

**Abstract**: Diabetic Retinopathy (DR) is a leading cause of vision loss around the world. To help diagnose it, numerous cutting-edge works have built powerful deep neural networks (DNNs) to automatically grade DR via retinal fundus images (RFIs). However, RFIs are commonly affected by camera exposure issues that may lead to incorrect grades. The mis-graded results can potentially pose high risks to an aggravation of the condition. In this paper, we study this problem from the viewpoint of adversarial attacks. We identify and introduce a novel solution to an entirely new task, termed as adversarial exposure attack, which is able to produce natural exposure images and mislead the state-of-the-art DNNs. We validate our proposed method on a real-world public DR dataset with three DNNs, e.g., ResNet50, MobileNet, and EfficientNet, demonstrating that our method achieves high image quality and success rate in transferring the attacks. Our method reveals the potential threats to DNN-based automatic DR grading and would benefit the development of exposure-robust DR grading methods in the future.



## **14. Bias in the Mirror : Are LLMs opinions robust to their own adversarial attacks ?**

cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13517v1) [paper-pdf](http://arxiv.org/pdf/2410.13517v1)

**Authors**: Virgile Rennard, Christos Xypolopoulos, Michalis Vazirgiannis

**Abstract**: Large language models (LLMs) inherit biases from their training data and alignment processes, influencing their responses in subtle ways. While many studies have examined these biases, little work has explored their robustness during interactions. In this paper, we introduce a novel approach where two instances of an LLM engage in self-debate, arguing opposing viewpoints to persuade a neutral version of the model. Through this, we evaluate how firmly biases hold and whether models are susceptible to reinforcing misinformation or shifting to harmful viewpoints. Our experiments span multiple LLMs of varying sizes, origins, and languages, providing deeper insights into bias persistence and flexibility across linguistic and cultural contexts.



## **15. MirrorCheck: Efficient Adversarial Defense for Vision-Language Models**

cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2406.09250v2) [paper-pdf](http://arxiv.org/pdf/2406.09250v2)

**Authors**: Samar Fares, Klea Ziu, Toluwani Aremu, Nikita Durasov, Martin Takáč, Pascal Fua, Karthik Nandakumar, Ivan Laptev

**Abstract**: Vision-Language Models (VLMs) are becoming increasingly vulnerable to adversarial attacks as various novel attack strategies are being proposed against these models. While existing defenses excel in unimodal contexts, they currently fall short in safeguarding VLMs against adversarial threats. To mitigate this vulnerability, we propose a novel, yet elegantly simple approach for detecting adversarial samples in VLMs. Our method leverages Text-to-Image (T2I) models to generate images based on captions produced by target VLMs. Subsequently, we calculate the similarities of the embeddings of both input and generated images in the feature space to identify adversarial samples. Empirical evaluations conducted on different datasets validate the efficacy of our approach, outperforming baseline methods adapted from image classification domains. Furthermore, we extend our methodology to classification tasks, showcasing its adaptability and model-agnostic nature. Theoretical analyses and empirical findings also show the resilience of our approach against adaptive attacks, positioning it as an excellent defense mechanism for real-world deployment against adversarial threats.



## **16. Byzantine-Resilient Output Optimization of Multiagent via Self-Triggered Hybrid Detection Approach**

eess.SY

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13454v1) [paper-pdf](http://arxiv.org/pdf/2410.13454v1)

**Authors**: Chenhang Yan, Liping Yan, Yuezu Lv, Bolei Dong, Yuanqing Xia

**Abstract**: How to achieve precise distributed optimization despite unknown attacks, especially the Byzantine attacks, is one of the critical challenges for multiagent systems. This paper addresses a distributed resilient optimization for linear heterogeneous multi-agent systems faced with adversarial threats. We establish a framework aimed at realizing resilient optimization for continuous-time systems by incorporating a novel self-triggered hybrid detection approach. The proposed hybrid detection approach is able to identify attacks on neighbors using both error thresholds and triggering intervals, thereby optimizing the balance between effective attack detection and the reduction of excessive communication triggers. Through using an edge-based adaptive self-triggered approach, each agent can receive its neighbors' information and determine whether these information is valid. If any neighbor prove invalid, each normal agent will isolate that neighbor by disconnecting communication along that specific edge. Importantly, our adaptive algorithm guarantees the accuracy of the optimization solution even when an agent is isolated by its neighbors.



## **17. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.02240v3) [paper-pdf](http://arxiv.org/pdf/2410.02240v3)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.



## **18. Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning**

cs.CL

16 pages, 5 figures

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13274v1) [paper-pdf](http://arxiv.org/pdf/2410.13274v1)

**Authors**: Minseok Choi, ChaeHun Park, Dohyun Lee, Jaegul Choo

**Abstract**: Large language models (LLMs) serve as giant information stores, often including personal or copyrighted data, and retraining them from scratch is not a viable option. This has led to the development of various fast, approximate unlearning techniques to selectively remove knowledge from LLMs. Prior research has largely focused on minimizing the probabilities of specific token sequences by reversing the language modeling objective. However, these methods still leave LLMs vulnerable to adversarial attacks that exploit indirect references. In this work, we examine the limitations of current unlearning techniques in effectively erasing a particular type of indirect prompt: multi-hop queries. Our findings reveal that existing methods fail to completely remove multi-hop knowledge when one of the intermediate hops is unlearned. To address this issue, we propose MUNCH, a simple uncertainty-based approach that breaks down multi-hop queries into subquestions and leverages the uncertainty of the unlearned model in final decision-making. Empirical results demonstrate the effectiveness of our framework, and MUNCH can be easily integrated with existing unlearning techniques, making it a flexible and useful solution for enhancing unlearning processes.



## **19. SPIN: Self-Supervised Prompt INjection**

cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13236v1) [paper-pdf](http://arxiv.org/pdf/2410.13236v1)

**Authors**: Leon Zhou, Junfeng Yang, Chengzhi Mao

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of important applications, yet their safety and reliability remain as major concerns. Various adversarial and jailbreak attacks have been proposed to bypass the safety alignment and cause the model to produce harmful responses. We introduce Self-supervised Prompt INjection (SPIN) which can detect and reverse these various attacks on LLMs. As our self-supervised prompt defense is done at inference-time, it is also compatible with existing alignment and adds an additional layer of safety for defense. Our benchmarks demonstrate that our system can reduce the attack success rate by up to 87.9%, while maintaining the performance on benign user requests. In addition, we discuss the situation of an adaptive attacker and show that our method is still resilient against attackers who are aware of our defense.



## **20. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

cs.CL

12 pages, 9 figures, EMNLP 2024 Findings

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2407.21659v4) [paper-pdf](http://arxiv.org/pdf/2407.21659v4)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.



## **21. Golyadkin's Torment: Doppelgängers and Adversarial Vulnerability**

cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13193v1) [paper-pdf](http://arxiv.org/pdf/2410.13193v1)

**Authors**: George I. Kamberov

**Abstract**: Many machine learning (ML) classifiers are claimed to outperform humans, but they still make mistakes that humans do not. The most notorious examples of such mistakes are adversarial visual metamers. This paper aims to define and investigate the phenomenon of adversarial Doppelgangers (AD), which includes adversarial visual metamers, and to compare the performance and robustness of ML classifiers to human performance.   We find that AD are inputs that are close to each other with respect to a perceptual metric defined in this paper. AD are qualitatively different from the usual adversarial examples. The vast majority of classifiers are vulnerable to AD and robustness-accuracy trade-offs may not improve them. Some classification problems may not admit any AD robust classifiers because the underlying classes are ambiguous. We provide criteria that can be used to determine whether a classification problem is well defined or not; describe the structure and attributes of an AD-robust classifier; introduce and explore the notions of conceptual entropy and regions of conceptual ambiguity for classifiers that are vulnerable to AD attacks, along with methods to bound the AD fooling rate of an attack. We define the notion of classifiers that exhibit hypersensitive behavior, that is, classifiers whose only mistakes are adversarial Doppelgangers. Improving the AD robustness of hyper-sensitive classifiers is equivalent to improving accuracy. We identify conditions guaranteeing that all classifiers with sufficiently high accuracy are hyper-sensitive.   Our findings are aimed at significant improvements in the reliability and security of machine learning systems.



## **22. Model Supply Chain Poisoning: Backdooring Pre-trained Models via Embedding Indistinguishability**

cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2401.15883v2) [paper-pdf](http://arxiv.org/pdf/2401.15883v2)

**Authors**: Hao Wang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang, Tao Xiang

**Abstract**: Pre-trained models (PTMs) are widely adopted across various downstream tasks in the machine learning supply chain. Adopting untrustworthy PTMs introduces significant security risks, where adversaries can poison the model supply chain by embedding hidden malicious behaviors (backdoors) into PTMs. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. This makes it challenging for the backdoors to persist and propagate through the supply chain. In this paper, we propose a novel and severer backdoor attack, TransTroj, which enables the backdoors embedded in PTMs to efficiently transfer in the model supply chain. In particular, we first formalize this attack as an indistinguishability problem between poisoned and clean samples in the embedding space. We decompose embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that our method significantly outperforms SOTA task-agnostic backdoor attacks -- achieving nearly 100\% attack success rate on most downstream tasks -- and demonstrates robustness under various system settings. Our findings underscore the urgent need to secure the model supply chain against such transferable backdoor attacks. The code is available at https://github.com/haowang-cqu/TransTroj .



## **23. Data Defenses Against Large Language Models**

cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13138v1) [paper-pdf](http://arxiv.org/pdf/2410.13138v1)

**Authors**: William Agnew, Harry H. Jiang, Cella Sum, Maarten Sap, Sauvik Das

**Abstract**: Large language models excel at performing inference over text to extract information, summarize information, or generate additional text. These inference capabilities are implicated in a variety of ethical harms spanning surveillance, labor displacement, and IP/copyright theft. While many policy, legal, and technical mitigations have been proposed to counteract these harms, these mitigations typically require cooperation from institutions that move slower than technical advances (i.e., governments) or that have few incentives to act to counteract these harms (i.e., the corporations that create and profit from these LLMs). In this paper, we define and build "data defenses" -- a novel strategy that directly empowers data owners to block LLMs from performing inference on their data. We create data defenses by developing a method to automatically generate adversarial prompt injections that, when added to input text, significantly reduce the ability of LLMs to accurately infer personally identifying information about the subject of the input text or to use copyrighted text in inference. We examine the ethics of enabling such direct resistance to LLM inference, and argue that making data defenses that resist and subvert LLMs enables the realization of important values such as data ownership, data sovereignty, and democratic control over AI systems. We verify that our data defenses are cheap and fast to generate, work on the latest commercial and open-source LLMs, resistance to countermeasures, and are robust to several different attack settings. Finally, we consider the security implications of LLM data defenses and outline several future research directions in this area. Our code is available at https://github.com/wagnew3/LLMDataDefenses and a tool for using our defenses to protect text against LLM inference is at https://wagnew3.github.io/LLM-Data-Defenses/.



## **24. Degraded Polygons Raise Fundamental Questions of Neural Network Perception**

cs.CV

Accepted as a conference paper to NeurIPS 2023 (Datasets & Benchmarks  Track)

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2306.04955v2) [paper-pdf](http://arxiv.org/pdf/2306.04955v2)

**Authors**: Leonard Tang, Dan Ley

**Abstract**: It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deep learning vision models suffer in a variety of settings that humans capably handle. In light of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images under degradation, first introduced over 30 years ago in the Recognition-by-Components theory of human vision. Specifically, we study the performance and behavior of neural networks on the seemingly simple task of classifying regular polygons at varying orders of degradation along their perimeters. To this end, we implement the Automated Shape Recoverability Test for rapidly generating large-scale datasets of perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity of neural networks to recognize and recover such degraded shapes when initialized with different priors. Ultimately, we find that neural networks' behavior on this simple task conflicts with human behavior, raising a fundamental question of the robustness and learning capabilities of modern computer vision models.



## **25. Hiding-in-Plain-Sight (HiPS) Attack on CLIP for Targetted Object Removal from Images**

cs.LG

Published in the 3rd Workshop on New Frontiers in Adversarial Machine  Learning at NeurIPS 2024. 10 pages, 7 figures, 3 tables

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.13010v1) [paper-pdf](http://arxiv.org/pdf/2410.13010v1)

**Authors**: Arka Daw, Megan Hong-Thanh Chung, Maria Mahbub, Amir Sadovnik

**Abstract**: Machine learning models are known to be vulnerable to adversarial attacks, but traditional attacks have mostly focused on single-modalities. With the rise of large multi-modal models (LMMs) like CLIP, which combine vision and language capabilities, new vulnerabilities have emerged. However, prior work in multimodal targeted attacks aim to completely change the model's output to what the adversary wants. In many realistic scenarios, an adversary might seek to make only subtle modifications to the output, so that the changes go unnoticed by downstream models or even by humans. We introduce Hiding-in-Plain-Sight (HiPS) attacks, a novel class of adversarial attacks that subtly modifies model predictions by selectively concealing target object(s), as if the target object was absent from the scene. We propose two HiPS attack variants, HiPS-cls and HiPS-cap, and demonstrate their effectiveness in transferring to downstream image captioning models, such as CLIP-Cap, for targeted object removal from image captions.



## **26. TMI! Finetuned Models Leak Private Information from their Pretraining Data**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2306.01181v3) [paper-pdf](http://arxiv.org/pdf/2306.01181v3)

**Authors**: John Abascal, Stanley Wu, Alina Oprea, Jonathan Ullman

**Abstract**: Transfer learning has become an increasingly popular technique in machine learning as a way to leverage a pretrained model trained for one task to assist with building a finetuned model for a related task. This paradigm has been especially popular for $\textit{privacy}$ in machine learning, where the pretrained model is considered public, and only the data for finetuning is considered sensitive. However, there are reasons to believe that the data used for pretraining is still sensitive, making it essential to understand how much information the finetuned model leaks about the pretraining data. In this work we propose a new membership-inference threat model where the adversary only has access to the finetuned model and would like to infer the membership of the pretraining data. To realize this threat model, we implement a novel metaclassifier-based attack, $\textbf{TMI}$, that leverages the influence of memorized pretraining samples on predictions in the downstream task. We evaluate $\textbf{TMI}$ on both vision and natural language tasks across multiple transfer learning settings, including finetuning with differential privacy. Through our evaluation, we find that $\textbf{TMI}$ can successfully infer membership of pretraining examples using query access to the finetuned model. An open-source implementation of $\textbf{TMI}$ can be found on GitHub: https://github.com/johnmath/tmi-pets24.



## **27. Adversarial Training of Two-Layer Polynomial and ReLU Activation Networks via Convex Optimization**

cs.LG

17 pages, 2 figures. Added a proof of the main theorem in the  appendix. Expanded numerical results section. Added references

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2405.14033v2) [paper-pdf](http://arxiv.org/pdf/2405.14033v2)

**Authors**: Daniel Kuelbs, Sanjay Lall, Mert Pilanci

**Abstract**: Training neural networks which are robust to adversarial attacks remains an important problem in deep learning, especially as heavily overparameterized models are adopted in safety-critical settings. Drawing from recent work which reformulates the training problems for two-layer ReLU and polynomial activation networks as convex programs, we devise a convex semidefinite program (SDP) for adversarial training of two-layer polynomial activation networks and prove that the convex SDP achieves the same globally optimal solution as its nonconvex counterpart. The convex SDP is observed to improve robust test accuracy against $\ell_\infty$ attacks relative to the original convex training formulation on multiple datasets. Additionally, we present scalable implementations of adversarial training for two-layer polynomial and ReLU networks which are compatible with standard machine learning libraries and GPU acceleration. Leveraging these implementations, we retrain the final two fully connected layers of a Pre-Activation ResNet-18 model on the CIFAR-10 dataset with both polynomial and ReLU activations. The two `robustified' models achieve significantly higher robust test accuracies against $\ell_\infty$ attacks than a Pre-Activation ResNet-18 model trained with sharpness-aware minimization, demonstrating the practical utility of convex adversarial training on large-scale problems.



## **28. Unitary Multi-Margin BERT for Robust Natural Language Processing**

cs.CL

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12759v1) [paper-pdf](http://arxiv.org/pdf/2410.12759v1)

**Authors**: Hao-Yuan Chang, Kang L. Wang

**Abstract**: Recent developments in adversarial attacks on deep learning leave many mission-critical natural language processing (NLP) systems at risk of exploitation. To address the lack of computationally efficient adversarial defense methods, this paper reports a novel, universal technique that drastically improves the robustness of Bidirectional Encoder Representations from Transformers (BERT) by combining the unitary weights with the multi-margin loss. We discover that the marriage of these two simple ideas amplifies the protection against malicious interference. Our model, the unitary multi-margin BERT (UniBERT), boosts post-attack classification accuracies significantly by 5.3% to 73.8% while maintaining competitive pre-attack accuracies. Furthermore, the pre-attack and post-attack accuracy tradeoff can be adjusted via a single scalar parameter to best fit the design requirements for the target applications.



## **29. ToBlend: Token-Level Blending With an Ensemble of LLMs to Attack AI-Generated Text Detection**

cs.CL

Submitted to ARR Oct-2024 Cycle

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2402.11167v2) [paper-pdf](http://arxiv.org/pdf/2402.11167v2)

**Authors**: Fan Huang, Haewoon Kwak, Jisun An

**Abstract**: The robustness of AI-content detection models against sophisticated adversarial strategies, such as paraphrasing or word switching, is a rising concern in natural language generation (NLG) applications. This study proposes ToBlend, a novel token-level ensemble text generation method to challenge the robustness of current AI-content detection approaches by utilizing multiple sets of candidate generative large language models (LLMs). By randomly sampling token(s) from candidate LLMs sets, we find ToBlend significantly drops the performance of most mainstream AI-content detection methods. We evaluate the text quality produced under different ToBlend settings based on annotations from experienced human experts. We proposed a fine-tuned Llama3.1 model to distinguish the ToBlend generated text more accurately. Our findings underscore our proposed text generation approach's great potential in deceiving and improving detection models. Our datasets, codes, and annotations are open-sourced.



## **30. Low-Rank Adversarial PGD Attack**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12607v1) [paper-pdf](http://arxiv.org/pdf/2410.12607v1)

**Authors**: Dayana Savostianova, Emanuele Zangrando, Francesco Tudisco

**Abstract**: Adversarial attacks on deep neural network models have seen rapid development and are extensively used to study the stability of these networks. Among various adversarial strategies, Projected Gradient Descent (PGD) is a widely adopted method in computer vision due to its effectiveness and quick implementation, making it suitable for adversarial training. In this work, we observe that in many cases, the perturbations computed using PGD predominantly affect only a portion of the singular value spectrum of the original image, suggesting that these perturbations are approximately low-rank. Motivated by this observation, we propose a variation of PGD that efficiently computes a low-rank attack. We extensively validate our method on a range of standard models as well as robust models that have undergone adversarial training. Our analysis indicates that the proposed low-rank PGD can be effectively used in adversarial training due to its straightforward and fast implementation coupled with competitive performance. Notably, we find that low-rank PGD often performs comparably to, and sometimes even outperforms, the traditional full-rank PGD attack, while using significantly less memory.



## **31. Efficient and Effective Universal Adversarial Attack against Vision-Language Pre-training Models**

cs.CV

11 pages

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.11639v2) [paper-pdf](http://arxiv.org/pdf/2410.11639v2)

**Authors**: Fan Yang, Yihao Huang, Kailong Wang, Ling Shi, Geguang Pu, Yang Liu, Haoyu Wang

**Abstract**: Vision-language pre-training (VLP) models, trained on large-scale image-text pairs, have become widely used across a variety of downstream vision-and-language (V+L) tasks. This widespread adoption raises concerns about their vulnerability to adversarial attacks. Non-universal adversarial attacks, while effective, are often impractical for real-time online applications due to their high computational demands per data instance. Recently, universal adversarial perturbations (UAPs) have been introduced as a solution, but existing generator-based UAP methods are significantly time-consuming. To overcome the limitation, we propose a direct optimization-based UAP approach, termed DO-UAP, which significantly reduces resource consumption while maintaining high attack performance. Specifically, we explore the necessity of multimodal loss design and introduce a useful data augmentation strategy. Extensive experiments conducted on three benchmark VLP datasets, six popular VLP models, and three classical downstream tasks demonstrate the efficiency and effectiveness of DO-UAP. Specifically, our approach drastically decreases the time consumption by 23-fold while achieving a better attack performance.



## **32. A Proactive Decoy Selection Scheme for Cyber Deception using MITRE ATT&CK**

cs.CR

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2404.12783v3) [paper-pdf](http://arxiv.org/pdf/2404.12783v3)

**Authors**: Marco Zambianco, Claudio Facchinetti, Domenico Siracusa

**Abstract**: Cyber deception allows compensating the late response of defenders countermeasures to the ever evolving tactics, techniques, and procedures (TTPs) of attackers. This proactive defense strategy employs decoys resembling legitimate system components to lure stealthy attackers within the defender environment, slowing and/or denying the accomplishment of their goals. In this regard, the selection of decoys that can expose the techniques used by malicious users plays a central role to incentivize their engagement. However, this is a difficult task to achieve in practice, since it requires an accurate and realistic modeling of the attacker capabilities and his possible targets. In this work, we tackle this challenge and we design a decoy selection scheme that is supported by an adversarial modeling based on empirical observation of real-world attackers. We take advantage of a domain-specific threat modelling language using MITRE ATT&CK framework as source of attacker TTPs targeting enterprise systems. In detail, we extract the information about the execution preconditions of each technique as well as its possible effects on the environment to generate attack graphs modeling the adversary capabilities. Based on this, we formulate a graph partition problem that minimizes the number of decoys detecting a corresponding number of techniques employed in various attack paths directed to specific targets. We compare our optimization-based decoy selection approach against several benchmark schemes that ignore the preconditions between the various attack steps. Results reveal that the proposed scheme provides the highest interception rate of attack paths using the lowest amount of decoys.



## **33. Query Provenance Analysis: Efficient and Robust Defense against Query-based Black-box Attacks**

cs.CR

The final version of this paper is going to appear in IEEE Symposium  on Security and Privacy 2025

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2405.20641v2) [paper-pdf](http://arxiv.org/pdf/2405.20641v2)

**Authors**: Shaofei Li, Ziqi Zhang, Haomin Jia, Ding Li, Yao Guo, Xiangqun Chen

**Abstract**: Query-based black-box attacks have emerged as a significant threat to machine learning systems, where adversaries can manipulate the input queries to generate adversarial examples that can cause misclassification of the model. To counter these attacks, researchers have proposed Stateful Defense Models (SDMs) for detecting adversarial query sequences and rejecting queries that are "similar" to the history queries. Existing state-of-the-art (SOTA) SDMs (e.g., BlackLight and PIHA) have shown great effectiveness in defending against these attacks. However, recent studies have shown that they are vulnerable to Oracle-guided Adaptive Rejection Sampling (OARS) attacks, which is a stronger adaptive attack strategy. It can be easily integrated with existing attack algorithms to evade the SDMs by generating queries with fine-tuned direction and step size of perturbations utilizing the leaked decision information from the SDMs.   In this paper, we propose a novel approach, Query Provenance Analysis (QPA), for more robust and efficient SDMs. QPA encapsulates the historical relationships among queries as the sequence feature to capture the fundamental difference between benign and adversarial query sequences. To utilize the query provenance, we propose an efficient query provenance analysis algorithm with dynamic management. We evaluate QPA compared with two baselines, BlackLight and PIHA, on four widely used datasets with six query-based black-box attack algorithms. The results show that QPA outperforms the baselines in terms of defense effectiveness and efficiency on both non-adaptive and adaptive attacks. Specifically, QPA reduces the Attack Success Rate (ASR) of OARS to 4.08%, comparing to 77.63% and 87.72% for BlackLight and PIHA, respectively. Moreover, QPA also achieves 7.67x and 2.25x higher throughput than BlackLight and PIHA.



## **34. Perseus: Leveraging Common Data Patterns with Curriculum Learning for More Robust Graph Neural Networks**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12425v1) [paper-pdf](http://arxiv.org/pdf/2410.12425v1)

**Authors**: Kaiwen Xia, Huijun Wu, Duanyu Li, Min Xie, Ruibo Wang, Wenzhe Zhang

**Abstract**: Graph Neural Networks (GNNs) excel at handling graph data but remain vulnerable to adversarial attacks. Existing defense methods typically rely on assumptions like graph sparsity and homophily to either preprocess the graph or guide structure learning. However, preprocessing methods often struggle to accurately distinguish between normal edges and adversarial perturbations, leading to suboptimal results due to the loss of valuable edge information. Robust graph neural network models train directly on graph data affected by adversarial perturbations, without preprocessing. This can cause the model to get stuck in poor local optima, negatively affecting its performance. To address these challenges, we propose Perseus, a novel adversarial defense method based on curriculum learning. Perseus assesses edge difficulty using global homophily and applies a curriculum learning strategy to adjust the learning order, guiding the model to learn the full graph structure while adaptively focusing on common data patterns. This approach mitigates the impact of adversarial perturbations. Experiments show that models trained with Perseus achieve superior performance and are significantly more robust to adversarial attacks.



## **35. DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12307v1) [paper-pdf](http://arxiv.org/pdf/2410.12307v1)

**Authors**: Fengpeng Li, Kemou Li, Haiwei Wu, Jinyu Tian, Jiantao Zhou

**Abstract**: To protect deep neural networks (DNNs) from adversarial attacks, adversarial training (AT) is developed by incorporating adversarial examples (AEs) into model training. Recent studies show that adversarial attacks disproportionately impact the patterns within the phase of the sample's frequency spectrum -- typically containing crucial semantic information -- more than those in the amplitude, resulting in the model's erroneous categorization of AEs. We find that, by mixing the amplitude of training samples' frequency spectrum with those of distractor images for AT, the model can be guided to focus on phase patterns unaffected by adversarial perturbations. As a result, the model's robustness can be improved. Unfortunately, it is still challenging to select appropriate distractor images, which should mix the amplitude without affecting the phase patterns. To this end, in this paper, we propose an optimized Adversarial Amplitude Generator (AAG) to achieve a better tradeoff between improving the model's robustness and retaining phase patterns. Based on this generator, together with an efficient AE production procedure, we design a new Dual Adversarial Training (DAT) strategy. Experiments on various datasets show that our proposed DAT leads to significantly improved robustness against diverse adversarial attacks.



## **36. MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2402.02263v5) [paper-pdf](http://arxiv.org/pdf/2402.02263v5)

**Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi

**Abstract**: Adversarial robustness often comes at the cost of degraded accuracy, impeding real-life applications of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.



## **37. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

cs.CR

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.13077v2) [paper-pdf](http://arxiv.org/pdf/2405.13077v2)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find that IRIS achieves jailbreak success rates of 98% on GPT-4, 92% on GPT-4 Turbo, and 94% on Llama-3.1-70B in under 7 queries. It significantly outperforms prior approaches in automatic, black-box, and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.



## **38. Taking off the Rose-Tinted Glasses: A Critical Look at Adversarial ML Through the Lens of Evasion Attacks**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.12076v1) [paper-pdf](http://arxiv.org/pdf/2410.12076v1)

**Authors**: Kevin Eykholt, Farhan Ahmed, Pratik Vaishnavi, Amir Rahmati

**Abstract**: The vulnerability of machine learning models in adversarial scenarios has garnered significant interest in the academic community over the past decade, resulting in a myriad of attacks and defenses. However, while the community appears to be overtly successful in devising new attacks across new contexts, the development of defenses has stalled. After a decade of research, we appear no closer to securing AI applications beyond additional training. Despite a lack of effective mitigations, AI development and its incorporation into existing systems charge full speed ahead with the rise of generative AI and large language models. Will our ineffectiveness in developing solutions to adversarial threats further extend to these new technologies?   In this paper, we argue that overly permissive attack and overly restrictive defensive threat models have hampered defense development in the ML domain. Through the lens of adversarial evasion attacks against neural networks, we critically examine common attack assumptions, such as the ability to bypass any defense not explicitly built into the model. We argue that these flawed assumptions, seen as reasonable by the community based on paper acceptance, have encouraged the development of adversarial attacks that map poorly to real-world scenarios. In turn, new defenses evaluated against these very attacks are inadvertently required to be almost perfect and incorporated as part of the model. But do they need to? In practice, machine learning models are deployed as a small component of a larger system. We analyze adversarial machine learning from a system security perspective rather than an AI perspective and its implications for emerging AI paradigms.



## **39. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11782v1) [paper-pdf](http://arxiv.org/pdf/2410.11782v1)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **40. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.20485v2) [paper-pdf](http://arxiv.org/pdf/2405.20485v2)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama, and show that they transfer to GPT-3.5 Turbo and GPT-4. Finally, we successfully conducted a Phantom attack on NVIDIA's black-box production RAG system, "Chat with RTX".



## **41. Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor**

cs.CR

Accepted by NeurIPS 2024. 32 pages, 7 figures, 28 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.16112v2) [paper-pdf](http://arxiv.org/pdf/2405.16112v2)

**Authors**: Shaokui Wei, Hongyuan Zha, Baoyuan Wu

**Abstract**: Data-poisoning backdoor attacks are serious security threats to machine learning models, where an adversary can manipulate the training dataset to inject backdoors into models. In this paper, we focus on in-training backdoor defense, aiming to train a clean model even when the dataset may be potentially poisoned. Unlike most existing methods that primarily detect and remove/unlearn suspicious samples to mitigate malicious backdoor attacks, we propose a novel defense approach called PDB (Proactive Defensive Backdoor). Specifically, PDB leverages the home-field advantage of defenders by proactively injecting a defensive backdoor into the model during training. Taking advantage of controlling the training process, the defensive backdoor is designed to suppress the malicious backdoor effectively while remaining secret to attackers. In addition, we introduce a reversible mapping to determine the defensive target label. During inference, PDB embeds a defensive trigger in the inputs and reverses the model's prediction, suppressing malicious backdoor and ensuring the model's utility on the original task. Experimental results across various datasets and models demonstrate that our approach achieves state-of-the-art defense performance against a wide range of backdoor attacks. The code is available at https://github.com/shawkui/Proactive_Defensive_Backdoor.



## **42. Security of and by Generative AI platforms**

cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.13899v1) [paper-pdf](http://arxiv.org/pdf/2410.13899v1)

**Authors**: Hari Hayagreevan, Souvik Khamaru

**Abstract**: This whitepaper highlights the dual importance of securing generative AI (genAI) platforms and leveraging genAI for cybersecurity. As genAI technologies proliferate, their misuse poses significant risks, including data breaches, model tampering, and malicious content generation. Securing these platforms is critical to protect sensitive data, ensure model integrity, and prevent adversarial attacks. Simultaneously, genAI presents opportunities for enhancing security by automating threat detection, vulnerability analysis, and incident response. The whitepaper explores strategies for robust security frameworks around genAI systems, while also showcasing how genAI can empower organizations to anticipate, detect, and mitigate sophisticated cyber threats.



## **43. GSE: Group-wise Sparse and Explainable Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2311.17434v2) [paper-pdf](http://arxiv.org/pdf/2311.17434v2)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, often regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. We address this by presenting a two-phase algorithm that generates group-wise sparse attacks within semantically meaningful areas of an image. Initially, we optimize a quasinorm adversarial loss using the $1/2-$quasinorm proximal operator tailored for non-convex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2-$norm regularization applied to perturbation magnitudes. Rigorous evaluations on CIFAR-10 and ImageNet datasets demonstrate a remarkable increase in group-wise sparsity, e.g., $50.9\%$ on CIFAR-10 and $38.4\%$ on ImageNet (average case, targeted attack). This performance improvement is accompanied by significantly faster computation times, improved explainability, and a $100\%$ attack success rate.



## **44. Information Importance-Aware Defense against Adversarial Attack for Automatic Modulation Classification:An XAI-Based Approach**

eess.SP

Accepted by WCSP 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11608v1) [paper-pdf](http://arxiv.org/pdf/2410.11608v1)

**Authors**: Jingchun Wang, Peihao Dong, Fuhui Zhou, Qihui Wu

**Abstract**: Deep learning (DL) has significantly improved automatic modulation classification (AMC) by leveraging neural networks as the feature extractor.However, as the DL-based AMC becomes increasingly widespread, it is faced with the severe secure issue from various adversarial attacks. Existing defense methods often suffer from the high computational cost, intractable parameter tuning, and insufficient robustness.This paper proposes an eXplainable artificial intelligence (XAI) defense approach, which uncovers the negative information caused by the adversarial attack through measuring the importance of input features based on the SHapley Additive exPlanations (SHAP).By properly removing the negative information in adversarial samples and then fine-tuning(FT) the model, the impact of the attacks on the classification result can be mitigated.Experimental results demonstrate that the proposed SHAP-FT improves the classification performance of the model by 15%-20% under different attack levels,which not only enhances model robustness against various attack levels but also reduces the resource consumption, validating its effectiveness in safeguarding communication networks.



## **45. RAUCA: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation**

cs.CV

12 pages. In Proceedings of the Forty-first International Conference  on Machine Learning (ICML), Vienna, Austria, July 21-27, 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2402.15853v2) [paper-pdf](http://arxiv.org/pdf/2402.15853v2)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle, resulting in suboptimal attack performance. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, Neural Renderer Plus (NRP), which can accurately project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA consistently outperforms existing methods in both simulation and real-world settings.



## **46. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11317v1) [paper-pdf](http://arxiv.org/pdf/2410.11317v1)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **47. On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning**

cs.LG

19 pages, 4 figures, 8 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.04682v2) [paper-pdf](http://arxiv.org/pdf/2410.04682v2)

**Authors**: Yongyi Su, Yushu Li, Nanqing Liu, Kui Jia, Xulei Yang, Chuan-Sheng Foo, Xun Xu

**Abstract**: Test-time adaptation (TTA) updates the model weights during the inference stage using testing data to enhance generalization. However, this practice exposes TTA to adversarial risks. Existing studies have shown that when TTA is updated with crafted adversarial test samples, also known as test-time poisoned data, the performance on benign samples can deteriorate. Nonetheless, the perceived adversarial risk may be overstated if the poisoned data is generated under overly strong assumptions. In this work, we first review realistic assumptions for test-time data poisoning, including white-box versus grey-box attacks, access to benign data, attack budget, and more. We then propose an effective and realistic attack method that better produces poisoned samples without access to benign samples, and derive an effective in-distribution attack objective. We also design two TTA-aware attack objectives. Our benchmarks of existing attack methods reveal that the TTA methods are more robust than previously believed. In addition, we analyze effective defense strategies to help develop adversarially robust TTA methods.



## **48. BRC20 Pinning Attack**

cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11295v1) [paper-pdf](http://arxiv.org/pdf/2410.11295v1)

**Authors**: Minfeng Qi, Qin Wang, Zhipeng Wang, Lin Zhong, Tianqing Zhu, Shiping Chen, William Knottenbelt

**Abstract**: BRC20 tokens are a type of non-fungible asset on the Bitcoin network. They allow users to embed customized content within Bitcoin satoshis. The related token frenzy has reached a market size of USD 3,650b over the past year (2023Q3-2024Q3). However, this intuitive design has not undergone serious security scrutiny.   We present the first in-depth analysis of the BRC20 transfer mechanism and identify a critical attack vector. A typical BRC20 transfer involves two bundled on-chain transactions with different fee levels: the first (i.e., Tx1) with a lower fee inscribes the transfer request, while the second (i.e., Tx2) with a higher fee finalizes the actual transfer. We find that an adversary can exploit this by sending a manipulated fee transaction (falling between the two fee levels), which allows Tx1 to be processed while Tx2 remains pinned in the mempool. This locks the BRC20 liquidity and disrupts normal transfers for users. We term this BRC20 pinning attack.   Our attack exposes an inherent design flaw that can be applied to 90+% inscription-based tokens within the Bitcoin ecosystem.   We also conducted the attack on Binance's ORDI hot wallet (the most prevalent BRC20 token and the most active wallet), resulting in a temporary suspension of ORDI withdrawals on Binance for 3.5 hours, which were shortly resumed after our communication.



## **49. Cognitive Overload Attack:Prompt Injection for Long Context**

cs.CL

40 pages, 31 Figures

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11272v1) [paper-pdf](http://arxiv.org/pdf/2410.11272v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Amin Karbasi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in performing tasks across various domains without needing explicit retraining. This capability, known as In-Context Learning (ICL), while impressive, exposes LLMs to a variety of adversarial prompts and jailbreaks that manipulate safety-trained LLMs into generating undesired or harmful output. In this paper, we propose a novel interpretation of ICL in LLMs through the lens of cognitive neuroscience, by drawing parallels between learning in human cognition with ICL. We applied the principles of Cognitive Load Theory in LLMs and empirically validate that similar to human cognition, LLMs also suffer from cognitive overload a state where the demand on cognitive processing exceeds the available capacity of the model, leading to potential errors. Furthermore, we demonstrated how an attacker can exploit ICL to jailbreak LLMs through deliberately designed prompts that induce cognitive overload on LLMs, thereby compromising the safety mechanisms of LLMs. We empirically validate this threat model by crafting various cognitive overload prompts and show that advanced models such as GPT-4, Claude-3.5 Sonnet, Claude-3 OPUS, Llama-3-70B-Instruct, Gemini-1.0-Pro, and Gemini-1.5-Pro can be successfully jailbroken, with attack success rates of up to 99.99%. Our findings highlight critical vulnerabilities in LLMs and underscore the urgency of developing robust safeguards. We propose integrating insights from cognitive load theory into the design and evaluation of LLMs to better anticipate and mitigate the risks of adversarial attacks. By expanding our experiments to encompass a broader range of models and by highlighting vulnerabilities in LLMs' ICL, we aim to ensure the development of safer and more reliable AI systems.



## **50. A Formal Framework for Assessing and Mitigating Emergent Security Risks in Generative AI Models: Bridging Theory and Dynamic Risk Mitigation**

cs.CR

This paper was accepted in NeurIPS 2024 workshop on Red Teaming  GenAI: What can we learn with Adversaries?

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.13897v1) [paper-pdf](http://arxiv.org/pdf/2410.13897v1)

**Authors**: Aviral Srivastava, Sourav Panda

**Abstract**: As generative AI systems, including large language models (LLMs) and diffusion models, advance rapidly, their growing adoption has led to new and complex security risks often overlooked in traditional AI risk assessment frameworks. This paper introduces a novel formal framework for categorizing and mitigating these emergent security risks by integrating adaptive, real-time monitoring, and dynamic risk mitigation strategies tailored to generative models' unique vulnerabilities. We identify previously under-explored risks, including latent space exploitation, multi-modal cross-attack vectors, and feedback-loop-induced model degradation. Our framework employs a layered approach, incorporating anomaly detection, continuous red-teaming, and real-time adversarial simulation to mitigate these risks. We focus on formal verification methods to ensure model robustness and scalability in the face of evolving threats. Though theoretical, this work sets the stage for future empirical validation by establishing a detailed methodology and metrics for evaluating the performance of risk mitigation strategies in generative AI systems. This framework addresses existing gaps in AI safety, offering a comprehensive road map for future research and implementation.



