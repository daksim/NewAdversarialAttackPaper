# Latest Adversarial Attack Papers
**update at 2024-10-19 09:52:15**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Multi-style conversion for semantic segmentation of lesions in fundus images by adversarial attacks**

cs.CV

preprint

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13822v1) [paper-pdf](http://arxiv.org/pdf/2410.13822v1)

**Authors**: Clément Playout, Renaud Duval, Marie Carole Boucher, Farida Cheriet

**Abstract**: The diagnosis of diabetic retinopathy, which relies on fundus images, faces challenges in achieving transparency and interpretability when using a global classification approach. However, segmentation-based databases are significantly more expensive to acquire and combining them is often problematic. This paper introduces a novel method, termed adversarial style conversion, to address the lack of standardization in annotation styles across diverse databases. By training a single architecture on combined databases, the model spontaneously modifies its segmentation style depending on the input, demonstrating the ability to convert among different labeling styles. The proposed methodology adds a linear probe to detect dataset origin based on encoder features and employs adversarial attacks to condition the model's segmentation style. Results indicate significant qualitative and quantitative through dataset combination, offering avenues for improved model generalization, uncertainty estimation and continuous interpolation between annotation styles. Our approach enables training a segmentation model with diverse databases while controlling and leveraging annotation styles for improved retinopathy diagnosis.



## **2. Persistent Pre-Training Poisoning of LLMs**

cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13722v1) [paper-pdf](http://arxiv.org/pdf/2410.13722v1)

**Authors**: Yiming Zhang, Javier Rando, Ivan Evtimov, Jianfeng Chi, Eric Michael Smith, Nicholas Carlini, Florian Tramèr, Daphne Ippolito

**Abstract**: Large language models are pre-trained on uncurated text datasets consisting of trillions of tokens scraped from the Web. Prior work has shown that: (1) web-scraped pre-training datasets can be practically poisoned by malicious actors; and (2) adversaries can compromise language models after poisoning fine-tuning datasets. Our work evaluates for the first time whether language models can also be compromised during pre-training, with a focus on the persistence of pre-training attacks after models are fine-tuned as helpful and harmless chatbots (i.e., after SFT and DPO). We pre-train a series of LLMs from scratch to measure the impact of a potential poisoning adversary under four different attack objectives (denial-of-service, belief manipulation, jailbreaking, and prompt stealing), and across a wide range of model sizes (from 600M to 7B). Our main result is that poisoning only 0.1% of a model's pre-training dataset is sufficient for three out of four attacks to measurably persist through post-training. Moreover, simple attacks like denial-of-service persist through post-training with a poisoning rate of only 0.001%.



## **3. Optimal MEV Extraction Using Absolute Commitments**

cs.GT

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13624v1) [paper-pdf](http://arxiv.org/pdf/2410.13624v1)

**Authors**: Daji Landis, Nikolaj I. Schwartzbach

**Abstract**: We propose a new, more potent attack on decentralized exchanges. This attack leverages absolute commitments, which are commitments that can condition on the strategies made by other agents. This attack allows an adversary to charge monopoly prices by committing to undercut those other miners that refuse to charge an even higher fee. This allows the miner to extract the maximum possible price from the user, potentially through side channels that evade the inefficiencies and fees usually incurred. This is considerably more efficient than the prevailing strategy of `sandwich attacks', wherein the adversary induces and profits from fluctuations in the market price to the detriment of users. The attack we propose can, in principle, be realized by the irrevocable and self-executing nature of smart contracts, which are readily available on many major blockchains. Thus, the attack could potentially be used against a decentralized exchange and could drastically reduce the utility of the affected exchange.



## **4. Transformer-Based Approaches for Sensor-Based Human Activity Recognition: Opportunities and Challenges**

cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13605v1) [paper-pdf](http://arxiv.org/pdf/2410.13605v1)

**Authors**: Clayton Souza Leite, Henry Mauranen, Aziza Zhanabatyrova, Yu Xiao

**Abstract**: Transformers have excelled in natural language processing and computer vision, paving their way to sensor-based Human Activity Recognition (HAR). Previous studies show that transformers outperform their counterparts exclusively when they harness abundant data or employ compute-intensive optimization algorithms. However, neither of these scenarios is viable in sensor-based HAR due to the scarcity of data in this field and the frequent need to perform training and inference on resource-constrained devices. Our extensive investigation into various implementations of transformer-based versus non-transformer-based HAR using wearable sensors, encompassing more than 500 experiments, corroborates these concerns. We observe that transformer-based solutions pose higher computational demands, consistently yield inferior performance, and experience significant performance degradation when quantized to accommodate resource-constrained devices. Additionally, transformers demonstrate lower robustness to adversarial attacks, posing a potential threat to user trust in HAR.



## **5. Adversarial Exposure Attack on Diabetic Retinopathy Imagery Grading**

cs.CV

13 pages, 7 figures

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2009.09231v2) [paper-pdf](http://arxiv.org/pdf/2009.09231v2)

**Authors**: Yupeng Cheng, Qing Guo, Felix Juefei-Xu, Huazhu Fu, Shang-Wei Lin, Weisi Lin

**Abstract**: Diabetic Retinopathy (DR) is a leading cause of vision loss around the world. To help diagnose it, numerous cutting-edge works have built powerful deep neural networks (DNNs) to automatically grade DR via retinal fundus images (RFIs). However, RFIs are commonly affected by camera exposure issues that may lead to incorrect grades. The mis-graded results can potentially pose high risks to an aggravation of the condition. In this paper, we study this problem from the viewpoint of adversarial attacks. We identify and introduce a novel solution to an entirely new task, termed as adversarial exposure attack, which is able to produce natural exposure images and mislead the state-of-the-art DNNs. We validate our proposed method on a real-world public DR dataset with three DNNs, e.g., ResNet50, MobileNet, and EfficientNet, demonstrating that our method achieves high image quality and success rate in transferring the attacks. Our method reveals the potential threats to DNN-based automatic DR grading and would benefit the development of exposure-robust DR grading methods in the future.



## **6. Bias in the Mirror : Are LLMs opinions robust to their own adversarial attacks ?**

cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13517v1) [paper-pdf](http://arxiv.org/pdf/2410.13517v1)

**Authors**: Virgile Rennard, Christos Xypolopoulos, Michalis Vazirgiannis

**Abstract**: Large language models (LLMs) inherit biases from their training data and alignment processes, influencing their responses in subtle ways. While many studies have examined these biases, little work has explored their robustness during interactions. In this paper, we introduce a novel approach where two instances of an LLM engage in self-debate, arguing opposing viewpoints to persuade a neutral version of the model. Through this, we evaluate how firmly biases hold and whether models are susceptible to reinforcing misinformation or shifting to harmful viewpoints. Our experiments span multiple LLMs of varying sizes, origins, and languages, providing deeper insights into bias persistence and flexibility across linguistic and cultural contexts.



## **7. MirrorCheck: Efficient Adversarial Defense for Vision-Language Models**

cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2406.09250v2) [paper-pdf](http://arxiv.org/pdf/2406.09250v2)

**Authors**: Samar Fares, Klea Ziu, Toluwani Aremu, Nikita Durasov, Martin Takáč, Pascal Fua, Karthik Nandakumar, Ivan Laptev

**Abstract**: Vision-Language Models (VLMs) are becoming increasingly vulnerable to adversarial attacks as various novel attack strategies are being proposed against these models. While existing defenses excel in unimodal contexts, they currently fall short in safeguarding VLMs against adversarial threats. To mitigate this vulnerability, we propose a novel, yet elegantly simple approach for detecting adversarial samples in VLMs. Our method leverages Text-to-Image (T2I) models to generate images based on captions produced by target VLMs. Subsequently, we calculate the similarities of the embeddings of both input and generated images in the feature space to identify adversarial samples. Empirical evaluations conducted on different datasets validate the efficacy of our approach, outperforming baseline methods adapted from image classification domains. Furthermore, we extend our methodology to classification tasks, showcasing its adaptability and model-agnostic nature. Theoretical analyses and empirical findings also show the resilience of our approach against adaptive attacks, positioning it as an excellent defense mechanism for real-world deployment against adversarial threats.



## **8. Byzantine-Resilient Output Optimization of Multiagent via Self-Triggered Hybrid Detection Approach**

eess.SY

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13454v1) [paper-pdf](http://arxiv.org/pdf/2410.13454v1)

**Authors**: Chenhang Yan, Liping Yan, Yuezu Lv, Bolei Dong, Yuanqing Xia

**Abstract**: How to achieve precise distributed optimization despite unknown attacks, especially the Byzantine attacks, is one of the critical challenges for multiagent systems. This paper addresses a distributed resilient optimization for linear heterogeneous multi-agent systems faced with adversarial threats. We establish a framework aimed at realizing resilient optimization for continuous-time systems by incorporating a novel self-triggered hybrid detection approach. The proposed hybrid detection approach is able to identify attacks on neighbors using both error thresholds and triggering intervals, thereby optimizing the balance between effective attack detection and the reduction of excessive communication triggers. Through using an edge-based adaptive self-triggered approach, each agent can receive its neighbors' information and determine whether these information is valid. If any neighbor prove invalid, each normal agent will isolate that neighbor by disconnecting communication along that specific edge. Importantly, our adaptive algorithm guarantees the accuracy of the optimization solution even when an agent is isolated by its neighbors.



## **9. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.02240v3) [paper-pdf](http://arxiv.org/pdf/2410.02240v3)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.



## **10. Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning**

cs.CL

16 pages, 5 figures

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13274v1) [paper-pdf](http://arxiv.org/pdf/2410.13274v1)

**Authors**: Minseok Choi, ChaeHun Park, Dohyun Lee, Jaegul Choo

**Abstract**: Large language models (LLMs) serve as giant information stores, often including personal or copyrighted data, and retraining them from scratch is not a viable option. This has led to the development of various fast, approximate unlearning techniques to selectively remove knowledge from LLMs. Prior research has largely focused on minimizing the probabilities of specific token sequences by reversing the language modeling objective. However, these methods still leave LLMs vulnerable to adversarial attacks that exploit indirect references. In this work, we examine the limitations of current unlearning techniques in effectively erasing a particular type of indirect prompt: multi-hop queries. Our findings reveal that existing methods fail to completely remove multi-hop knowledge when one of the intermediate hops is unlearned. To address this issue, we propose MUNCH, a simple uncertainty-based approach that breaks down multi-hop queries into subquestions and leverages the uncertainty of the unlearned model in final decision-making. Empirical results demonstrate the effectiveness of our framework, and MUNCH can be easily integrated with existing unlearning techniques, making it a flexible and useful solution for enhancing unlearning processes.



## **11. SPIN: Self-Supervised Prompt INjection**

cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13236v1) [paper-pdf](http://arxiv.org/pdf/2410.13236v1)

**Authors**: Leon Zhou, Junfeng Yang, Chengzhi Mao

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of important applications, yet their safety and reliability remain as major concerns. Various adversarial and jailbreak attacks have been proposed to bypass the safety alignment and cause the model to produce harmful responses. We introduce Self-supervised Prompt INjection (SPIN) which can detect and reverse these various attacks on LLMs. As our self-supervised prompt defense is done at inference-time, it is also compatible with existing alignment and adds an additional layer of safety for defense. Our benchmarks demonstrate that our system can reduce the attack success rate by up to 87.9%, while maintaining the performance on benign user requests. In addition, we discuss the situation of an adaptive attacker and show that our method is still resilient against attackers who are aware of our defense.



## **12. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

cs.CL

12 pages, 9 figures, EMNLP 2024 Findings

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2407.21659v4) [paper-pdf](http://arxiv.org/pdf/2407.21659v4)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.



## **13. Golyadkin's Torment: Doppelgängers and Adversarial Vulnerability**

cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13193v1) [paper-pdf](http://arxiv.org/pdf/2410.13193v1)

**Authors**: George I. Kamberov

**Abstract**: Many machine learning (ML) classifiers are claimed to outperform humans, but they still make mistakes that humans do not. The most notorious examples of such mistakes are adversarial visual metamers. This paper aims to define and investigate the phenomenon of adversarial Doppelgangers (AD), which includes adversarial visual metamers, and to compare the performance and robustness of ML classifiers to human performance.   We find that AD are inputs that are close to each other with respect to a perceptual metric defined in this paper. AD are qualitatively different from the usual adversarial examples. The vast majority of classifiers are vulnerable to AD and robustness-accuracy trade-offs may not improve them. Some classification problems may not admit any AD robust classifiers because the underlying classes are ambiguous. We provide criteria that can be used to determine whether a classification problem is well defined or not; describe the structure and attributes of an AD-robust classifier; introduce and explore the notions of conceptual entropy and regions of conceptual ambiguity for classifiers that are vulnerable to AD attacks, along with methods to bound the AD fooling rate of an attack. We define the notion of classifiers that exhibit hypersensitive behavior, that is, classifiers whose only mistakes are adversarial Doppelgangers. Improving the AD robustness of hyper-sensitive classifiers is equivalent to improving accuracy. We identify conditions guaranteeing that all classifiers with sufficiently high accuracy are hyper-sensitive.   Our findings are aimed at significant improvements in the reliability and security of machine learning systems.



## **14. Model Supply Chain Poisoning: Backdooring Pre-trained Models via Embedding Indistinguishability**

cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2401.15883v2) [paper-pdf](http://arxiv.org/pdf/2401.15883v2)

**Authors**: Hao Wang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang, Tao Xiang

**Abstract**: Pre-trained models (PTMs) are widely adopted across various downstream tasks in the machine learning supply chain. Adopting untrustworthy PTMs introduces significant security risks, where adversaries can poison the model supply chain by embedding hidden malicious behaviors (backdoors) into PTMs. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. This makes it challenging for the backdoors to persist and propagate through the supply chain. In this paper, we propose a novel and severer backdoor attack, TransTroj, which enables the backdoors embedded in PTMs to efficiently transfer in the model supply chain. In particular, we first formalize this attack as an indistinguishability problem between poisoned and clean samples in the embedding space. We decompose embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that our method significantly outperforms SOTA task-agnostic backdoor attacks -- achieving nearly 100\% attack success rate on most downstream tasks -- and demonstrates robustness under various system settings. Our findings underscore the urgent need to secure the model supply chain against such transferable backdoor attacks. The code is available at https://github.com/haowang-cqu/TransTroj .



## **15. Data Defenses Against Large Language Models**

cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13138v1) [paper-pdf](http://arxiv.org/pdf/2410.13138v1)

**Authors**: William Agnew, Harry H. Jiang, Cella Sum, Maarten Sap, Sauvik Das

**Abstract**: Large language models excel at performing inference over text to extract information, summarize information, or generate additional text. These inference capabilities are implicated in a variety of ethical harms spanning surveillance, labor displacement, and IP/copyright theft. While many policy, legal, and technical mitigations have been proposed to counteract these harms, these mitigations typically require cooperation from institutions that move slower than technical advances (i.e., governments) or that have few incentives to act to counteract these harms (i.e., the corporations that create and profit from these LLMs). In this paper, we define and build "data defenses" -- a novel strategy that directly empowers data owners to block LLMs from performing inference on their data. We create data defenses by developing a method to automatically generate adversarial prompt injections that, when added to input text, significantly reduce the ability of LLMs to accurately infer personally identifying information about the subject of the input text or to use copyrighted text in inference. We examine the ethics of enabling such direct resistance to LLM inference, and argue that making data defenses that resist and subvert LLMs enables the realization of important values such as data ownership, data sovereignty, and democratic control over AI systems. We verify that our data defenses are cheap and fast to generate, work on the latest commercial and open-source LLMs, resistance to countermeasures, and are robust to several different attack settings. Finally, we consider the security implications of LLM data defenses and outline several future research directions in this area. Our code is available at https://github.com/wagnew3/LLMDataDefenses and a tool for using our defenses to protect text against LLM inference is at https://wagnew3.github.io/LLM-Data-Defenses/.



## **16. Degraded Polygons Raise Fundamental Questions of Neural Network Perception**

cs.CV

Accepted as a conference paper to NeurIPS 2023 (Datasets & Benchmarks  Track)

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2306.04955v2) [paper-pdf](http://arxiv.org/pdf/2306.04955v2)

**Authors**: Leonard Tang, Dan Ley

**Abstract**: It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deep learning vision models suffer in a variety of settings that humans capably handle. In light of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images under degradation, first introduced over 30 years ago in the Recognition-by-Components theory of human vision. Specifically, we study the performance and behavior of neural networks on the seemingly simple task of classifying regular polygons at varying orders of degradation along their perimeters. To this end, we implement the Automated Shape Recoverability Test for rapidly generating large-scale datasets of perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity of neural networks to recognize and recover such degraded shapes when initialized with different priors. Ultimately, we find that neural networks' behavior on this simple task conflicts with human behavior, raising a fundamental question of the robustness and learning capabilities of modern computer vision models.



## **17. Hiding-in-Plain-Sight (HiPS) Attack on CLIP for Targetted Object Removal from Images**

cs.LG

Published in the 3rd Workshop on New Frontiers in Adversarial Machine  Learning at NeurIPS 2024. 10 pages, 7 figures, 3 tables

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.13010v1) [paper-pdf](http://arxiv.org/pdf/2410.13010v1)

**Authors**: Arka Daw, Megan Hong-Thanh Chung, Maria Mahbub, Amir Sadovnik

**Abstract**: Machine learning models are known to be vulnerable to adversarial attacks, but traditional attacks have mostly focused on single-modalities. With the rise of large multi-modal models (LMMs) like CLIP, which combine vision and language capabilities, new vulnerabilities have emerged. However, prior work in multimodal targeted attacks aim to completely change the model's output to what the adversary wants. In many realistic scenarios, an adversary might seek to make only subtle modifications to the output, so that the changes go unnoticed by downstream models or even by humans. We introduce Hiding-in-Plain-Sight (HiPS) attacks, a novel class of adversarial attacks that subtly modifies model predictions by selectively concealing target object(s), as if the target object was absent from the scene. We propose two HiPS attack variants, HiPS-cls and HiPS-cap, and demonstrate their effectiveness in transferring to downstream image captioning models, such as CLIP-Cap, for targeted object removal from image captions.



## **18. TMI! Finetuned Models Leak Private Information from their Pretraining Data**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2306.01181v3) [paper-pdf](http://arxiv.org/pdf/2306.01181v3)

**Authors**: John Abascal, Stanley Wu, Alina Oprea, Jonathan Ullman

**Abstract**: Transfer learning has become an increasingly popular technique in machine learning as a way to leverage a pretrained model trained for one task to assist with building a finetuned model for a related task. This paradigm has been especially popular for $\textit{privacy}$ in machine learning, where the pretrained model is considered public, and only the data for finetuning is considered sensitive. However, there are reasons to believe that the data used for pretraining is still sensitive, making it essential to understand how much information the finetuned model leaks about the pretraining data. In this work we propose a new membership-inference threat model where the adversary only has access to the finetuned model and would like to infer the membership of the pretraining data. To realize this threat model, we implement a novel metaclassifier-based attack, $\textbf{TMI}$, that leverages the influence of memorized pretraining samples on predictions in the downstream task. We evaluate $\textbf{TMI}$ on both vision and natural language tasks across multiple transfer learning settings, including finetuning with differential privacy. Through our evaluation, we find that $\textbf{TMI}$ can successfully infer membership of pretraining examples using query access to the finetuned model. An open-source implementation of $\textbf{TMI}$ can be found on GitHub: https://github.com/johnmath/tmi-pets24.



## **19. Adversarial Training of Two-Layer Polynomial and ReLU Activation Networks via Convex Optimization**

cs.LG

17 pages, 2 figures. Added a proof of the main theorem in the  appendix. Expanded numerical results section. Added references

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2405.14033v2) [paper-pdf](http://arxiv.org/pdf/2405.14033v2)

**Authors**: Daniel Kuelbs, Sanjay Lall, Mert Pilanci

**Abstract**: Training neural networks which are robust to adversarial attacks remains an important problem in deep learning, especially as heavily overparameterized models are adopted in safety-critical settings. Drawing from recent work which reformulates the training problems for two-layer ReLU and polynomial activation networks as convex programs, we devise a convex semidefinite program (SDP) for adversarial training of two-layer polynomial activation networks and prove that the convex SDP achieves the same globally optimal solution as its nonconvex counterpart. The convex SDP is observed to improve robust test accuracy against $\ell_\infty$ attacks relative to the original convex training formulation on multiple datasets. Additionally, we present scalable implementations of adversarial training for two-layer polynomial and ReLU networks which are compatible with standard machine learning libraries and GPU acceleration. Leveraging these implementations, we retrain the final two fully connected layers of a Pre-Activation ResNet-18 model on the CIFAR-10 dataset with both polynomial and ReLU activations. The two `robustified' models achieve significantly higher robust test accuracies against $\ell_\infty$ attacks than a Pre-Activation ResNet-18 model trained with sharpness-aware minimization, demonstrating the practical utility of convex adversarial training on large-scale problems.



## **20. Unitary Multi-Margin BERT for Robust Natural Language Processing**

cs.CL

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12759v1) [paper-pdf](http://arxiv.org/pdf/2410.12759v1)

**Authors**: Hao-Yuan Chang, Kang L. Wang

**Abstract**: Recent developments in adversarial attacks on deep learning leave many mission-critical natural language processing (NLP) systems at risk of exploitation. To address the lack of computationally efficient adversarial defense methods, this paper reports a novel, universal technique that drastically improves the robustness of Bidirectional Encoder Representations from Transformers (BERT) by combining the unitary weights with the multi-margin loss. We discover that the marriage of these two simple ideas amplifies the protection against malicious interference. Our model, the unitary multi-margin BERT (UniBERT), boosts post-attack classification accuracies significantly by 5.3% to 73.8% while maintaining competitive pre-attack accuracies. Furthermore, the pre-attack and post-attack accuracy tradeoff can be adjusted via a single scalar parameter to best fit the design requirements for the target applications.



## **21. ToBlend: Token-Level Blending With an Ensemble of LLMs to Attack AI-Generated Text Detection**

cs.CL

Submitted to ARR Oct-2024 Cycle

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2402.11167v2) [paper-pdf](http://arxiv.org/pdf/2402.11167v2)

**Authors**: Fan Huang, Haewoon Kwak, Jisun An

**Abstract**: The robustness of AI-content detection models against sophisticated adversarial strategies, such as paraphrasing or word switching, is a rising concern in natural language generation (NLG) applications. This study proposes ToBlend, a novel token-level ensemble text generation method to challenge the robustness of current AI-content detection approaches by utilizing multiple sets of candidate generative large language models (LLMs). By randomly sampling token(s) from candidate LLMs sets, we find ToBlend significantly drops the performance of most mainstream AI-content detection methods. We evaluate the text quality produced under different ToBlend settings based on annotations from experienced human experts. We proposed a fine-tuned Llama3.1 model to distinguish the ToBlend generated text more accurately. Our findings underscore our proposed text generation approach's great potential in deceiving and improving detection models. Our datasets, codes, and annotations are open-sourced.



## **22. Low-Rank Adversarial PGD Attack**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12607v1) [paper-pdf](http://arxiv.org/pdf/2410.12607v1)

**Authors**: Dayana Savostianova, Emanuele Zangrando, Francesco Tudisco

**Abstract**: Adversarial attacks on deep neural network models have seen rapid development and are extensively used to study the stability of these networks. Among various adversarial strategies, Projected Gradient Descent (PGD) is a widely adopted method in computer vision due to its effectiveness and quick implementation, making it suitable for adversarial training. In this work, we observe that in many cases, the perturbations computed using PGD predominantly affect only a portion of the singular value spectrum of the original image, suggesting that these perturbations are approximately low-rank. Motivated by this observation, we propose a variation of PGD that efficiently computes a low-rank attack. We extensively validate our method on a range of standard models as well as robust models that have undergone adversarial training. Our analysis indicates that the proposed low-rank PGD can be effectively used in adversarial training due to its straightforward and fast implementation coupled with competitive performance. Notably, we find that low-rank PGD often performs comparably to, and sometimes even outperforms, the traditional full-rank PGD attack, while using significantly less memory.



## **23. Efficient and Effective Universal Adversarial Attack against Vision-Language Pre-training Models**

cs.CV

11 pages

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.11639v2) [paper-pdf](http://arxiv.org/pdf/2410.11639v2)

**Authors**: Fan Yang, Yihao Huang, Kailong Wang, Ling Shi, Geguang Pu, Yang Liu, Haoyu Wang

**Abstract**: Vision-language pre-training (VLP) models, trained on large-scale image-text pairs, have become widely used across a variety of downstream vision-and-language (V+L) tasks. This widespread adoption raises concerns about their vulnerability to adversarial attacks. Non-universal adversarial attacks, while effective, are often impractical for real-time online applications due to their high computational demands per data instance. Recently, universal adversarial perturbations (UAPs) have been introduced as a solution, but existing generator-based UAP methods are significantly time-consuming. To overcome the limitation, we propose a direct optimization-based UAP approach, termed DO-UAP, which significantly reduces resource consumption while maintaining high attack performance. Specifically, we explore the necessity of multimodal loss design and introduce a useful data augmentation strategy. Extensive experiments conducted on three benchmark VLP datasets, six popular VLP models, and three classical downstream tasks demonstrate the efficiency and effectiveness of DO-UAP. Specifically, our approach drastically decreases the time consumption by 23-fold while achieving a better attack performance.



## **24. A Proactive Decoy Selection Scheme for Cyber Deception using MITRE ATT&CK**

cs.CR

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2404.12783v3) [paper-pdf](http://arxiv.org/pdf/2404.12783v3)

**Authors**: Marco Zambianco, Claudio Facchinetti, Domenico Siracusa

**Abstract**: Cyber deception allows compensating the late response of defenders countermeasures to the ever evolving tactics, techniques, and procedures (TTPs) of attackers. This proactive defense strategy employs decoys resembling legitimate system components to lure stealthy attackers within the defender environment, slowing and/or denying the accomplishment of their goals. In this regard, the selection of decoys that can expose the techniques used by malicious users plays a central role to incentivize their engagement. However, this is a difficult task to achieve in practice, since it requires an accurate and realistic modeling of the attacker capabilities and his possible targets. In this work, we tackle this challenge and we design a decoy selection scheme that is supported by an adversarial modeling based on empirical observation of real-world attackers. We take advantage of a domain-specific threat modelling language using MITRE ATT&CK framework as source of attacker TTPs targeting enterprise systems. In detail, we extract the information about the execution preconditions of each technique as well as its possible effects on the environment to generate attack graphs modeling the adversary capabilities. Based on this, we formulate a graph partition problem that minimizes the number of decoys detecting a corresponding number of techniques employed in various attack paths directed to specific targets. We compare our optimization-based decoy selection approach against several benchmark schemes that ignore the preconditions between the various attack steps. Results reveal that the proposed scheme provides the highest interception rate of attack paths using the lowest amount of decoys.



## **25. Query Provenance Analysis: Efficient and Robust Defense against Query-based Black-box Attacks**

cs.CR

The final version of this paper is going to appear in IEEE Symposium  on Security and Privacy 2025

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2405.20641v2) [paper-pdf](http://arxiv.org/pdf/2405.20641v2)

**Authors**: Shaofei Li, Ziqi Zhang, Haomin Jia, Ding Li, Yao Guo, Xiangqun Chen

**Abstract**: Query-based black-box attacks have emerged as a significant threat to machine learning systems, where adversaries can manipulate the input queries to generate adversarial examples that can cause misclassification of the model. To counter these attacks, researchers have proposed Stateful Defense Models (SDMs) for detecting adversarial query sequences and rejecting queries that are "similar" to the history queries. Existing state-of-the-art (SOTA) SDMs (e.g., BlackLight and PIHA) have shown great effectiveness in defending against these attacks. However, recent studies have shown that they are vulnerable to Oracle-guided Adaptive Rejection Sampling (OARS) attacks, which is a stronger adaptive attack strategy. It can be easily integrated with existing attack algorithms to evade the SDMs by generating queries with fine-tuned direction and step size of perturbations utilizing the leaked decision information from the SDMs.   In this paper, we propose a novel approach, Query Provenance Analysis (QPA), for more robust and efficient SDMs. QPA encapsulates the historical relationships among queries as the sequence feature to capture the fundamental difference between benign and adversarial query sequences. To utilize the query provenance, we propose an efficient query provenance analysis algorithm with dynamic management. We evaluate QPA compared with two baselines, BlackLight and PIHA, on four widely used datasets with six query-based black-box attack algorithms. The results show that QPA outperforms the baselines in terms of defense effectiveness and efficiency on both non-adaptive and adaptive attacks. Specifically, QPA reduces the Attack Success Rate (ASR) of OARS to 4.08%, comparing to 77.63% and 87.72% for BlackLight and PIHA, respectively. Moreover, QPA also achieves 7.67x and 2.25x higher throughput than BlackLight and PIHA.



## **26. Perseus: Leveraging Common Data Patterns with Curriculum Learning for More Robust Graph Neural Networks**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12425v1) [paper-pdf](http://arxiv.org/pdf/2410.12425v1)

**Authors**: Kaiwen Xia, Huijun Wu, Duanyu Li, Min Xie, Ruibo Wang, Wenzhe Zhang

**Abstract**: Graph Neural Networks (GNNs) excel at handling graph data but remain vulnerable to adversarial attacks. Existing defense methods typically rely on assumptions like graph sparsity and homophily to either preprocess the graph or guide structure learning. However, preprocessing methods often struggle to accurately distinguish between normal edges and adversarial perturbations, leading to suboptimal results due to the loss of valuable edge information. Robust graph neural network models train directly on graph data affected by adversarial perturbations, without preprocessing. This can cause the model to get stuck in poor local optima, negatively affecting its performance. To address these challenges, we propose Perseus, a novel adversarial defense method based on curriculum learning. Perseus assesses edge difficulty using global homophily and applies a curriculum learning strategy to adjust the learning order, guiding the model to learn the full graph structure while adaptively focusing on common data patterns. This approach mitigates the impact of adversarial perturbations. Experiments show that models trained with Perseus achieve superior performance and are significantly more robust to adversarial attacks.



## **27. DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12307v1) [paper-pdf](http://arxiv.org/pdf/2410.12307v1)

**Authors**: Fengpeng Li, Kemou Li, Haiwei Wu, Jinyu Tian, Jiantao Zhou

**Abstract**: To protect deep neural networks (DNNs) from adversarial attacks, adversarial training (AT) is developed by incorporating adversarial examples (AEs) into model training. Recent studies show that adversarial attacks disproportionately impact the patterns within the phase of the sample's frequency spectrum -- typically containing crucial semantic information -- more than those in the amplitude, resulting in the model's erroneous categorization of AEs. We find that, by mixing the amplitude of training samples' frequency spectrum with those of distractor images for AT, the model can be guided to focus on phase patterns unaffected by adversarial perturbations. As a result, the model's robustness can be improved. Unfortunately, it is still challenging to select appropriate distractor images, which should mix the amplitude without affecting the phase patterns. To this end, in this paper, we propose an optimized Adversarial Amplitude Generator (AAG) to achieve a better tradeoff between improving the model's robustness and retaining phase patterns. Based on this generator, together with an efficient AE production procedure, we design a new Dual Adversarial Training (DAT) strategy. Experiments on various datasets show that our proposed DAT leads to significantly improved robustness against diverse adversarial attacks.



## **28. MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**

cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2402.02263v5) [paper-pdf](http://arxiv.org/pdf/2402.02263v5)

**Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi

**Abstract**: Adversarial robustness often comes at the cost of degraded accuracy, impeding real-life applications of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.



## **29. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

cs.CR

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.13077v2) [paper-pdf](http://arxiv.org/pdf/2405.13077v2)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find that IRIS achieves jailbreak success rates of 98% on GPT-4, 92% on GPT-4 Turbo, and 94% on Llama-3.1-70B in under 7 queries. It significantly outperforms prior approaches in automatic, black-box, and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.



## **30. Taking off the Rose-Tinted Glasses: A Critical Look at Adversarial ML Through the Lens of Evasion Attacks**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.12076v1) [paper-pdf](http://arxiv.org/pdf/2410.12076v1)

**Authors**: Kevin Eykholt, Farhan Ahmed, Pratik Vaishnavi, Amir Rahmati

**Abstract**: The vulnerability of machine learning models in adversarial scenarios has garnered significant interest in the academic community over the past decade, resulting in a myriad of attacks and defenses. However, while the community appears to be overtly successful in devising new attacks across new contexts, the development of defenses has stalled. After a decade of research, we appear no closer to securing AI applications beyond additional training. Despite a lack of effective mitigations, AI development and its incorporation into existing systems charge full speed ahead with the rise of generative AI and large language models. Will our ineffectiveness in developing solutions to adversarial threats further extend to these new technologies?   In this paper, we argue that overly permissive attack and overly restrictive defensive threat models have hampered defense development in the ML domain. Through the lens of adversarial evasion attacks against neural networks, we critically examine common attack assumptions, such as the ability to bypass any defense not explicitly built into the model. We argue that these flawed assumptions, seen as reasonable by the community based on paper acceptance, have encouraged the development of adversarial attacks that map poorly to real-world scenarios. In turn, new defenses evaluated against these very attacks are inadvertently required to be almost perfect and incorporated as part of the model. But do they need to? In practice, machine learning models are deployed as a small component of a larger system. We analyze adversarial machine learning from a system security perspective rather than an AI perspective and its implications for emerging AI paradigms.



## **31. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11782v1) [paper-pdf](http://arxiv.org/pdf/2410.11782v1)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **32. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.20485v2) [paper-pdf](http://arxiv.org/pdf/2405.20485v2)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama, and show that they transfer to GPT-3.5 Turbo and GPT-4. Finally, we successfully conducted a Phantom attack on NVIDIA's black-box production RAG system, "Chat with RTX".



## **33. Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor**

cs.CR

Accepted by NeurIPS 2024. 32 pages, 7 figures, 28 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.16112v2) [paper-pdf](http://arxiv.org/pdf/2405.16112v2)

**Authors**: Shaokui Wei, Hongyuan Zha, Baoyuan Wu

**Abstract**: Data-poisoning backdoor attacks are serious security threats to machine learning models, where an adversary can manipulate the training dataset to inject backdoors into models. In this paper, we focus on in-training backdoor defense, aiming to train a clean model even when the dataset may be potentially poisoned. Unlike most existing methods that primarily detect and remove/unlearn suspicious samples to mitigate malicious backdoor attacks, we propose a novel defense approach called PDB (Proactive Defensive Backdoor). Specifically, PDB leverages the home-field advantage of defenders by proactively injecting a defensive backdoor into the model during training. Taking advantage of controlling the training process, the defensive backdoor is designed to suppress the malicious backdoor effectively while remaining secret to attackers. In addition, we introduce a reversible mapping to determine the defensive target label. During inference, PDB embeds a defensive trigger in the inputs and reverses the model's prediction, suppressing malicious backdoor and ensuring the model's utility on the original task. Experimental results across various datasets and models demonstrate that our approach achieves state-of-the-art defense performance against a wide range of backdoor attacks. The code is available at https://github.com/shawkui/Proactive_Defensive_Backdoor.



## **34. GSE: Group-wise Sparse and Explainable Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2311.17434v2) [paper-pdf](http://arxiv.org/pdf/2311.17434v2)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, often regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. We address this by presenting a two-phase algorithm that generates group-wise sparse attacks within semantically meaningful areas of an image. Initially, we optimize a quasinorm adversarial loss using the $1/2-$quasinorm proximal operator tailored for non-convex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2-$norm regularization applied to perturbation magnitudes. Rigorous evaluations on CIFAR-10 and ImageNet datasets demonstrate a remarkable increase in group-wise sparsity, e.g., $50.9\%$ on CIFAR-10 and $38.4\%$ on ImageNet (average case, targeted attack). This performance improvement is accompanied by significantly faster computation times, improved explainability, and a $100\%$ attack success rate.



## **35. Information Importance-Aware Defense against Adversarial Attack for Automatic Modulation Classification:An XAI-Based Approach**

eess.SP

Accepted by WCSP 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11608v1) [paper-pdf](http://arxiv.org/pdf/2410.11608v1)

**Authors**: Jingchun Wang, Peihao Dong, Fuhui Zhou, Qihui Wu

**Abstract**: Deep learning (DL) has significantly improved automatic modulation classification (AMC) by leveraging neural networks as the feature extractor.However, as the DL-based AMC becomes increasingly widespread, it is faced with the severe secure issue from various adversarial attacks. Existing defense methods often suffer from the high computational cost, intractable parameter tuning, and insufficient robustness.This paper proposes an eXplainable artificial intelligence (XAI) defense approach, which uncovers the negative information caused by the adversarial attack through measuring the importance of input features based on the SHapley Additive exPlanations (SHAP).By properly removing the negative information in adversarial samples and then fine-tuning(FT) the model, the impact of the attacks on the classification result can be mitigated.Experimental results demonstrate that the proposed SHAP-FT improves the classification performance of the model by 15%-20% under different attack levels,which not only enhances model robustness against various attack levels but also reduces the resource consumption, validating its effectiveness in safeguarding communication networks.



## **36. RAUCA: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation**

cs.CV

12 pages. In Proceedings of the Forty-first International Conference  on Machine Learning (ICML), Vienna, Austria, July 21-27, 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2402.15853v2) [paper-pdf](http://arxiv.org/pdf/2402.15853v2)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle, resulting in suboptimal attack performance. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, Neural Renderer Plus (NRP), which can accurately project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA consistently outperforms existing methods in both simulation and real-world settings.



## **37. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11317v1) [paper-pdf](http://arxiv.org/pdf/2410.11317v1)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **38. On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning**

cs.LG

19 pages, 4 figures, 8 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.04682v2) [paper-pdf](http://arxiv.org/pdf/2410.04682v2)

**Authors**: Yongyi Su, Yushu Li, Nanqing Liu, Kui Jia, Xulei Yang, Chuan-Sheng Foo, Xun Xu

**Abstract**: Test-time adaptation (TTA) updates the model weights during the inference stage using testing data to enhance generalization. However, this practice exposes TTA to adversarial risks. Existing studies have shown that when TTA is updated with crafted adversarial test samples, also known as test-time poisoned data, the performance on benign samples can deteriorate. Nonetheless, the perceived adversarial risk may be overstated if the poisoned data is generated under overly strong assumptions. In this work, we first review realistic assumptions for test-time data poisoning, including white-box versus grey-box attacks, access to benign data, attack budget, and more. We then propose an effective and realistic attack method that better produces poisoned samples without access to benign samples, and derive an effective in-distribution attack objective. We also design two TTA-aware attack objectives. Our benchmarks of existing attack methods reveal that the TTA methods are more robust than previously believed. In addition, we analyze effective defense strategies to help develop adversarially robust TTA methods.



## **39. BRC20 Pinning Attack**

cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11295v1) [paper-pdf](http://arxiv.org/pdf/2410.11295v1)

**Authors**: Minfeng Qi, Qin Wang, Zhipeng Wang, Lin Zhong, Tianqing Zhu, Shiping Chen, William Knottenbelt

**Abstract**: BRC20 tokens are a type of non-fungible asset on the Bitcoin network. They allow users to embed customized content within Bitcoin satoshis. The related token frenzy has reached a market size of USD 3,650b over the past year (2023Q3-2024Q3). However, this intuitive design has not undergone serious security scrutiny.   We present the first in-depth analysis of the BRC20 transfer mechanism and identify a critical attack vector. A typical BRC20 transfer involves two bundled on-chain transactions with different fee levels: the first (i.e., Tx1) with a lower fee inscribes the transfer request, while the second (i.e., Tx2) with a higher fee finalizes the actual transfer. We find that an adversary can exploit this by sending a manipulated fee transaction (falling between the two fee levels), which allows Tx1 to be processed while Tx2 remains pinned in the mempool. This locks the BRC20 liquidity and disrupts normal transfers for users. We term this BRC20 pinning attack.   Our attack exposes an inherent design flaw that can be applied to 90+% inscription-based tokens within the Bitcoin ecosystem.   We also conducted the attack on Binance's ORDI hot wallet (the most prevalent BRC20 token and the most active wallet), resulting in a temporary suspension of ORDI withdrawals on Binance for 3.5 hours, which were shortly resumed after our communication.



## **40. Cognitive Overload Attack:Prompt Injection for Long Context**

cs.CL

40 pages, 31 Figures

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11272v1) [paper-pdf](http://arxiv.org/pdf/2410.11272v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Amin Karbasi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in performing tasks across various domains without needing explicit retraining. This capability, known as In-Context Learning (ICL), while impressive, exposes LLMs to a variety of adversarial prompts and jailbreaks that manipulate safety-trained LLMs into generating undesired or harmful output. In this paper, we propose a novel interpretation of ICL in LLMs through the lens of cognitive neuroscience, by drawing parallels between learning in human cognition with ICL. We applied the principles of Cognitive Load Theory in LLMs and empirically validate that similar to human cognition, LLMs also suffer from cognitive overload a state where the demand on cognitive processing exceeds the available capacity of the model, leading to potential errors. Furthermore, we demonstrated how an attacker can exploit ICL to jailbreak LLMs through deliberately designed prompts that induce cognitive overload on LLMs, thereby compromising the safety mechanisms of LLMs. We empirically validate this threat model by crafting various cognitive overload prompts and show that advanced models such as GPT-4, Claude-3.5 Sonnet, Claude-3 OPUS, Llama-3-70B-Instruct, Gemini-1.0-Pro, and Gemini-1.5-Pro can be successfully jailbroken, with attack success rates of up to 99.99%. Our findings highlight critical vulnerabilities in LLMs and underscore the urgency of developing robust safeguards. We propose integrating insights from cognitive load theory into the design and evaluation of LLMs to better anticipate and mitigate the risks of adversarial attacks. By expanding our experiments to encompass a broader range of models and by highlighting vulnerabilities in LLMs' ICL, we aim to ensure the development of safer and more reliable AI systems.



## **41. Adversarially Guided Stateful Defense Against Backdoor Attacks in Federated Deep Learning**

cs.LG

16 pages, Accepted at ACSAC 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11205v1) [paper-pdf](http://arxiv.org/pdf/2410.11205v1)

**Authors**: Hassan Ali, Surya Nepal, Salil S. Kanhere, Sanjay Jha

**Abstract**: Recent works have shown that Federated Learning (FL) is vulnerable to backdoor attacks. Existing defenses cluster submitted updates from clients and select the best cluster for aggregation. However, they often rely on unrealistic assumptions regarding client submissions and sampled clients population while choosing the best cluster. We show that in realistic FL settings, state-of-the-art (SOTA) defenses struggle to perform well against backdoor attacks in FL. To address this, we highlight that backdoored submissions are adversarially biased and overconfident compared to clean submissions. We, therefore, propose an Adversarially Guided Stateful Defense (AGSD) against backdoor attacks on Deep Neural Networks (DNNs) in FL scenarios. AGSD employs adversarial perturbations to a small held-out dataset to compute a novel metric, called the trust index, that guides the cluster selection without relying on any unrealistic assumptions regarding client submissions. Moreover, AGSD maintains a trust state history of each client that adaptively penalizes backdoored clients and rewards clean clients. In realistic FL settings, where SOTA defenses mostly fail to resist attacks, AGSD mostly outperforms all SOTA defenses with minimal drop in clean accuracy (5% in the worst-case compared to best accuracy) even when (a) given a very small held-out dataset -- typically AGSD assumes 50 samples (<= 0.1% of the training data) and (b) no heldout dataset is available, and out-of-distribution data is used instead. For reproducibility, our code will be openly available at: https://github.com/hassanalikhatim/AGSD.



## **42. Fast Second-Order Online Kernel Learning through Incremental Matrix Sketching and Decomposition**

cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11188v1) [paper-pdf](http://arxiv.org/pdf/2410.11188v1)

**Authors**: Dongxie Wen, Xiao Zhang, Zhewei Wei

**Abstract**: Online Kernel Learning (OKL) has attracted considerable research interest due to its promising predictive performance in streaming environments. Second-order approaches are particularly appealing for OKL as they often offer substantial improvements in regret guarantees. However, existing second-order OKL approaches suffer from at least quadratic time complexity with respect to the pre-set budget, rendering them unsuitable for meeting the real-time demands of large-scale streaming recommender systems. The singular value decomposition required to obtain explicit feature mapping is also computationally expensive due to the complete decomposition process. Moreover, the absence of incremental updates to manage approximate kernel space causes these algorithms to perform poorly in adversarial environments and real-world streaming recommendation datasets. To address these issues, we propose FORKS, a fast incremental matrix sketching and decomposition approach tailored for second-order OKL. FORKS constructs an incremental maintenance paradigm for second-order kernelized gradient descent, which includes incremental matrix sketching for kernel approximation and incremental matrix decomposition for explicit feature mapping construction. Theoretical analysis demonstrates that FORKS achieves a logarithmic regret guarantee on par with other second-order approaches while maintaining a linear time complexity w.r.t. the budget, significantly enhancing efficiency over existing approaches. We validate the performance of FORKS through extensive experiments conducted on real-world streaming recommendation datasets, demonstrating its superior scalability and robustness against adversarial attacks.



## **43. Sensor Deprivation Attacks for Stealthy UAV Manipulation**

cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.11131v1) [paper-pdf](http://arxiv.org/pdf/2410.11131v1)

**Authors**: Alessandro Erba, John H. Castellanos, Sahil Sihag, Saman Zonouz, Nils Ole Tippenhauer

**Abstract**: Unmanned Aerial Vehicles autonomously perform tasks with the use of state-of-the-art control algorithms. These control algorithms rely on the freshness and correctness of sensor readings. Incorrect control actions lead to catastrophic destabilization of the process.   In this work, we propose a multi-part \emph{Sensor Deprivation Attacks} (SDAs), aiming to stealthily impact process control via sensor reconfiguration. In the first part, the attacker will inject messages on local buses that connect to the sensor. The injected message reconfigures the sensors, e.g.,~to suspend the sensing. In the second part, those manipulation primitives are selectively used to cause adversarial sensor values at the controller, transparently to the data consumer. In the third part, the manipulated sensor values lead to unwanted control actions (e.g. a drone crash). We experimentally investigate all three parts of our proposed attack. Our findings show that i)~reconfiguring sensors can have surprising effects on reported sensor values, and ii)~the attacker can stall the overall Kalman Filter state estimation, leading to a complete stop of control computations. As a result, the UAV becomes destabilized, leading to a crash or significant deviation from its planned trajectory (over 30 meters). We also propose an attack synthesis methodology that optimizes the timing of these SDA manipulations, maximizing their impact. Notably, our results demonstrate that these SDAs evade detection by state-of-the-art UAV anomaly detectors.   Our work shows that attacks on sensors are not limited to continuously inducing random measurements, and demonstrate that sensor reconfiguration can completely stall the drone controller. In our experiments, state-of-the-art UAV controller software and countermeasures are unable to handle such manipulations. Hence, we also discuss new corresponding countermeasures.



## **44. Denial-of-Service Poisoning Attacks against Large Language Models**

cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10760v1) [paper-pdf](http://arxiv.org/pdf/2410.10760v1)

**Authors**: Kuofeng Gao, Tianyu Pang, Chao Du, Yong Yang, Shu-Tao Xia, Min Lin

**Abstract**: Recent studies have shown that LLMs are vulnerable to denial-of-service (DoS) attacks, where adversarial inputs like spelling errors or non-semantic prompts trigger endless outputs without generating an [EOS] token. These attacks can potentially cause high latency and make LLM services inaccessible to other users or tasks. However, when there are speech-to-text interfaces (e.g., voice commands to a robot), executing such DoS attacks becomes challenging, as it is difficult to introduce spelling errors or non-semantic prompts through speech. A simple DoS attack in these scenarios would be to instruct the model to "Keep repeating Hello", but we observe that relying solely on natural instructions limits output length, which is bounded by the maximum length of the LLM's supervised finetuning (SFT) data. To overcome this limitation, we propose poisoning-based DoS (P-DoS) attacks for LLMs, demonstrating that injecting a single poisoned sample designed for DoS purposes can break the output length limit. For example, a poisoned sample can successfully attack GPT-4o and GPT-4o mini (via OpenAI's finetuning API) using less than $1, causing repeated outputs up to the maximum inference length (16K tokens, compared to 0.5K before poisoning). Additionally, we perform comprehensive ablation studies on open-source LLMs and extend our method to LLM agents, where attackers can control both the finetuning dataset and algorithm. Our findings underscore the urgent need for defenses against P-DoS attacks to secure LLMs. Our code is available at https://github.com/sail-sg/P-DoS.



## **45. Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings**

cs.LG

Code and pre-trained models are available at  https://github.com/AdaptiveMotorControlLab/AROS

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10744v1) [paper-pdf](http://arxiv.org/pdf/2410.10744v1)

**Authors**: Hossein Mirzaei, Mackenzie W. Mathis

**Abstract**: Despite significant advancements in out-of-distribution (OOD) detection, existing methods still struggle to maintain robustness against adversarial attacks, compromising their reliability in critical real-world applications. Previous studies have attempted to address this challenge by exposing detectors to auxiliary OOD datasets alongside adversarial training. However, the increased data complexity inherent in adversarial training, and the myriad of ways that OOD samples can arise during testing, often prevent these approaches from establishing robust decision boundaries. To address these limitations, we propose AROS, a novel approach leveraging neural ordinary differential equations (NODEs) with Lyapunov stability theorem in order to obtain robust embeddings for OOD detection. By incorporating a tailored loss function, we apply Lyapunov stability theory to ensure that both in-distribution (ID) and OOD data converge to stable equilibrium points within the dynamical system. This approach encourages any perturbed input to return to its stable equilibrium, thereby enhancing the model's robustness against adversarial perturbations. To not use additional data, we generate fake OOD embeddings by sampling from low-likelihood regions of the ID data feature space, approximating the boundaries where OOD data are likely to reside. To then further enhance robustness, we propose the use of an orthogonal binary layer following the stable feature space, which maximizes the separation between the equilibrium points of ID and OOD samples. We validate our method through extensive experiments across several benchmarks, demonstrating superior performance, particularly under adversarial attacks. Notably, our approach improves robust detection performance from 37.8% to 80.1% on CIFAR-10 vs. CIFAR-100 and from 29.0% to 67.0% on CIFAR-100 vs. CIFAR-10.



## **46. Towards Calibrated Losses for Adversarial Robust Reject Option Classification**

cs.LG

Accepted at Asian Conference on Machine Learning (ACML) , 2024

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10736v1) [paper-pdf](http://arxiv.org/pdf/2410.10736v1)

**Authors**: Vrund Shah, Tejas Chaudhari, Naresh Manwani

**Abstract**: Robustness towards adversarial attacks is a vital property for classifiers in several applications such as autonomous driving, medical diagnosis, etc. Also, in such scenarios, where the cost of misclassification is very high, knowing when to abstain from prediction becomes crucial. A natural question is which surrogates can be used to ensure learning in scenarios where the input points are adversarially perturbed and the classifier can abstain from prediction? This paper aims to characterize and design surrogates calibrated in "Adversarial Robust Reject Option" setting. First, we propose an adversarial robust reject option loss $\ell_{d}^{\gamma}$ and analyze it for the hypothesis set of linear classifiers ($\mathcal{H}_{\textrm{lin}}$). Next, we provide a complete characterization result for any surrogate to be $(\ell_{d}^{\gamma},\mathcal{H}_{\textrm{lin}})$- calibrated. To demonstrate the difficulty in designing surrogates to $\ell_{d}^{\gamma}$, we show negative calibration results for convex surrogates and quasi-concave conditional risk cases (these gave positive calibration in adversarial setting without reject option). We also empirically argue that Shifted Double Ramp Loss (DRL) and Shifted Double Sigmoid Loss (DSL) satisfy the calibration conditions. Finally, we demonstrate the robustness of shifted DRL and shifted DSL against adversarial perturbations on a synthetically generated dataset.



## **47. Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues**

cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10700v1) [paper-pdf](http://arxiv.org/pdf/2410.10700v1)

**Authors**: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao

**Abstract**: This study exposes the safety vulnerabilities of Large Language Models (LLMs) in multi-turn interactions, where malicious users can obscure harmful intents across several queries. We introduce ActorAttack, a novel multi-turn attack method inspired by actor-network theory, which models a network of semantically linked actors as attack clues to generate diverse and effective attack paths toward harmful targets. ActorAttack addresses two main challenges in multi-turn attacks: (1) concealing harmful intents by creating an innocuous conversation topic about the actor, and (2) uncovering diverse attack paths towards the same harmful target by leveraging LLMs' knowledge to specify the correlated actors as various attack clues. In this way, ActorAttack outperforms existing single-turn and multi-turn attack methods across advanced aligned LLMs, even for GPT-o1. We will publish a dataset called SafeMTData, which includes multi-turn adversarial prompts and safety alignment data, generated by ActorAttack. We demonstrate that models safety-tuned using our safety dataset are more robust to multi-turn attacks. Code is available at https://github.com/renqibing/ActorAttack.



## **48. Enhancing Robustness in Deep Reinforcement Learning: A Lyapunov Exponent Approach**

cs.LG

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10674v1) [paper-pdf](http://arxiv.org/pdf/2410.10674v1)

**Authors**: Rory Young, Nicolas Pugeault

**Abstract**: Deep reinforcement learning agents achieve state-of-the-art performance in a wide range of simulated control tasks. However, successful applications to real-world problems remain limited. One reason for this dichotomy is because the learned policies are not robust to observation noise or adversarial attacks. In this paper, we investigate the robustness of deep RL policies to a single small state perturbation in deterministic continuous control tasks. We demonstrate that RL policies can be deterministically chaotic as small perturbations to the system state have a large impact on subsequent state and reward trajectories. This unstable non-linear behaviour has two consequences: First, inaccuracies in sensor readings, or adversarial attacks, can cause significant performance degradation; Second, even policies that show robust performance in terms of rewards may have unpredictable behaviour in practice. These two facets of chaos in RL policies drastically restrict the application of deep RL to real-world problems. To address this issue, we propose an improvement on the successful Dreamer V3 architecture, implementing a Maximal Lyapunov Exponent regularisation. This new approach reduces the chaotic state dynamics, rendering the learnt policies more resilient to sensor noise or adversarial attacks and thereby improving the suitability of Deep Reinforcement Learning for real-world applications.



## **49. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

cs.LG

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10572v1) [paper-pdf](http://arxiv.org/pdf/2410.10572v1)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.



## **50. ROSAR: An Adversarial Re-Training Framework for Robust Side-Scan Sonar Object Detection**

cs.CV

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10554v1) [paper-pdf](http://arxiv.org/pdf/2410.10554v1)

**Authors**: Martin Aubard, László Antal, Ana Madureira, Luis F. Teixeira, Erika Ábrahám

**Abstract**: This paper introduces ROSAR, a novel framework enhancing the robustness of deep learning object detection models tailored for side-scan sonar (SSS) images, generated by autonomous underwater vehicles using sonar sensors. By extending our prior work on knowledge distillation (KD), this framework integrates KD with adversarial retraining to address the dual challenges of model efficiency and robustness against SSS noises. We introduce three novel, publicly available SSS datasets, capturing different sonar setups and noise conditions. We propose and formalize two SSS safety properties and utilize them to generate adversarial datasets for retraining. Through a comparative analysis of projected gradient descent (PGD) and patch-based adversarial attacks, ROSAR demonstrates significant improvements in model robustness and detection accuracy under SSS-specific conditions, enhancing the model's robustness by up to 1.85%. ROSAR is available at https://github.com/remaro-network/ROSAR-framework.



