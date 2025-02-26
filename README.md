# Latest Adversarial Attack Papers
**update at 2025-02-26 10:09:48**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Learning atomic forces from uncertainty-calibrated adversarial attacks**

physics.comp-ph

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18314v1) [paper-pdf](http://arxiv.org/pdf/2502.18314v1)

**Authors**: Henrique Musseli Cezar, Tilmann Bodenstein, Henrik Andersen Sveinsson, Morten Ledum, Simen Reine, Sigbjørn Løland Bore

**Abstract**: Adversarial approaches, which intentionally challenge machine learning models by generating difficult examples, are increasingly being adopted to improve machine learning interatomic potentials (MLIPs). While already providing great practical value, little is known about the actual prediction errors of MLIPs on adversarial structures and whether these errors can be controlled. We propose the Calibrated Adversarial Geometry Optimization (CAGO) algorithm to discover adversarial structures with user-assigned errors. Through uncertainty calibration, the estimated uncertainty of MLIPs is unified with real errors. By performing geometry optimization for calibrated uncertainty, we reach adversarial structures with the user-assigned target MLIP prediction error. Integrating with active learning pipelines, we benchmark CAGO, demonstrating stable MLIPs that systematically converge structural, dynamical, and thermodynamical properties for liquid water and water adsorption in a metal-organic framework within only hundreds of training structures, where previously many thousands were typically required.



## **2. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.01386v2) [paper-pdf](http://arxiv.org/pdf/2502.01386v2)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



## **3. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18176v1) [paper-pdf](http://arxiv.org/pdf/2502.18176v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.



## **4. Exploring the Robustness and Transferability of Patch-Based Adversarial Attacks in Quantized Neural Networks**

cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.15246v2) [paper-pdf](http://arxiv.org/pdf/2411.15246v2)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized neural networks (QNNs) are increasingly used for efficient deployment of deep learning models on resource-constrained platforms, such as mobile devices and edge computing systems. While quantization reduces model size and computational demands, its impact on adversarial robustness-especially against patch-based attacks-remains inadequately addressed. Patch-based attacks, characterized by localized, high-visibility perturbations, pose significant security risks due to their transferability and resilience. In this study, we systematically evaluate the vulnerability of QNNs to patch-based adversarial attacks across various quantization levels and architectures, focusing on factors that contribute to the robustness of these attacks. Through experiments analyzing feature representations, quantization strength, gradient alignment, and spatial sensitivity, we find that patch attacks consistently achieve high success rates across bitwidths and architectures, demonstrating significant transferability even in heavily quantized models. Contrary to the expectation that quantization might enhance adversarial defenses, our results show that QNNs remain highly susceptible to patch attacks due to the persistence of distinct, localized features within quantized representations. These findings underscore the need for quantization-aware defenses that address the specific challenges posed by patch-based attacks. Our work contributes to a deeper understanding of adversarial robustness in QNNs and aims to guide future research in developing secure, quantization-compatible defenses for real-world applications.



## **5. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13175v2) [paper-pdf](http://arxiv.org/pdf/2502.13175v2)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.



## **6. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2410.23091v7) [paper-pdf](http://arxiv.org/pdf/2410.23091v7)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.



## **7. Towards Certification of Uncertainty Calibration under Adversarial Attacks**

cs.LG

10 pages main paper, appendix included Published at: International  Conference on Learning Representations (ICLR) 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2405.13922v3) [paper-pdf](http://arxiv.org/pdf/2405.13922v3)

**Authors**: Cornelius Emde, Francesco Pinto, Thomas Lukasiewicz, Philip H. S. Torr, Adel Bibi

**Abstract**: Since neural classifiers are known to be sensitive to adversarial perturbations that alter their accuracy, \textit{certification methods} have been developed to provide provable guarantees on the insensitivity of their predictions to such perturbations. Furthermore, in safety-critical applications, the frequentist interpretation of the confidence of a classifier (also known as model calibration) can be of utmost importance. This property can be measured via the Brier score or the expected calibration error. We show that attacks can significantly harm calibration, and thus propose certified calibration as worst-case bounds on calibration under adversarial perturbations. Specifically, we produce analytic bounds for the Brier score and approximate bounds via the solution of a mixed-integer program on the expected calibration error. Finally, we propose novel calibration attacks and demonstrate how they can improve model calibration through \textit{adversarial calibration training}.



## **8. Model-Free Adversarial Purification via Coarse-To-Fine Tensor Network Representation**

cs.LG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17972v1) [paper-pdf](http://arxiv.org/pdf/2502.17972v1)

**Authors**: Guang Lin, Duc Thien Nguyen, Zerui Tao, Konstantinos Slavakis, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Deep neural networks are known to be vulnerable to well-designed adversarial attacks. Although numerous defense strategies have been proposed, many are tailored to the specific attacks or tasks and often fail to generalize across diverse scenarios. In this paper, we propose Tensor Network Purification (TNP), a novel model-free adversarial purification method by a specially designed tensor network decomposition algorithm. TNP depends neither on the pre-trained generative model nor the specific dataset, resulting in strong robustness across diverse adversarial scenarios. To this end, the key challenge lies in relaxing Gaussian-noise assumptions of classical decompositions and accommodating the unknown distribution of adversarial perturbations. Unlike the low-rank representation of classical decompositions, TNP aims to reconstruct the unobserved clean examples from an adversarial example. Specifically, TNP leverages progressive downsampling and introduces a novel adversarial optimization objective to address the challenge of minimizing reconstruction error but without inadvertently restoring adversarial perturbations. Extensive experiments conducted on CIFAR-10, CIFAR-100, and ImageNet demonstrate that our method generalizes effectively across various norm threats, attack types, and tasks, providing a versatile and promising adversarial purification technique.



## **9. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

cs.CY

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.12659v2) [paper-pdf](http://arxiv.org/pdf/2502.12659v2)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.



## **10. LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection**

cs.LG

PAKDD 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.09271v3) [paper-pdf](http://arxiv.org/pdf/2502.09271v3)

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.



## **11. Relationship between Uncertainty in DNNs and Adversarial Attacks**

cs.LG

review

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2409.13232v2) [paper-pdf](http://arxiv.org/pdf/2409.13232v2)

**Authors**: Mabel Ogonna, Abigail Adeniran, Adewale Adeyemo

**Abstract**: Deep Neural Networks (DNNs) have achieved state of the art results and even outperformed human accuracy in many challenging tasks, leading to DNNs adoption in a variety of fields including natural language processing, pattern recognition, prediction, and control optimization. However, DNNs are accompanied by uncertainty about their results, causing them to predict an outcome that is either incorrect or outside of a certain level of confidence. These uncertainties stem from model or data constraints, which could be exacerbated by adversarial attacks. Adversarial attacks aim to provide perturbed input to DNNs, causing the DNN to make incorrect predictions or increase model uncertainty. In this review, we explore the relationship between DNN uncertainty and adversarial attacks, emphasizing how adversarial attacks might raise DNN uncertainty.



## **12. The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence**

cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17420v1) [paper-pdf](http://arxiv.org/pdf/2502.17420v1)

**Authors**: Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger

**Abstract**: The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a single refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of representational independence that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.



## **13. Emoti-Attack: Zero-Perturbation Adversarial Attacks on NLP Systems via Emoji Sequences**

cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17392v1) [paper-pdf](http://arxiv.org/pdf/2502.17392v1)

**Authors**: Yangshijie Zhang

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success in the field of natural language processing (NLP), leading to widely recognized applications such as ChatGPT. However, the vulnerability of these models to adversarial attacks remains a significant concern. Unlike continuous domains like images, text exists in a discrete space, making even minor alterations at the sentence, word, or character level easily perceptible to humans. This inherent discreteness also complicates the use of conventional optimization techniques, as text is non-differentiable. Previous research on adversarial attacks in text has focused on character-level, word-level, sentence-level, and multi-level approaches, all of which suffer from inefficiency or perceptibility issues due to the need for multiple queries or significant semantic shifts.   In this work, we introduce a novel adversarial attack method, Emoji-Attack, which leverages the manipulation of emojis to create subtle, yet effective, perturbations. Unlike character- and word-level strategies, Emoji-Attack targets emojis as a distinct layer of attack, resulting in less noticeable changes with minimal disruption to the text. This approach has been largely unexplored in previous research, which typically focuses on emoji insertion as an extension of character-level attacks. Our experiments demonstrate that Emoji-Attack achieves strong attack performance on both large and small models, making it a promising technique for enhancing adversarial robustness in NLP systems.



## **14. On the Vulnerability of Concept Erasure in Diffusion Models**

cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17537v1) [paper-pdf](http://arxiv.org/pdf/2502.17537v1)

**Authors**: Lucas Beerens, Alex D. Richardson, Kaicheng Zhang, Dongdong Chen

**Abstract**: The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. To address these issues, research on machine unlearning has developed various concept erasure methods, which aim to remove the effect of unwanted data through post-hoc training. However, we show these erasure techniques are vulnerable, where images of supposedly erased concepts can still be generated using adversarially crafted prompts. We introduce RECORD, a coordinate-descent-based algorithm that discovers prompts capable of eliciting the generation of erased content. We demonstrate that RECORD significantly beats the attack success rate of current state-of-the-art attack methods. Furthermore, our findings reveal that models subjected to concept erasure are more susceptible to adversarial attacks than previously anticipated, highlighting the urgency for more robust unlearning approaches. We open source all our code at https://github.com/LucasBeerens/RECORD



## **15. Order Fairness Evaluation of DAG-based ledgers**

cs.CR

19 pages with 9 pages dedicated to references and appendices, 23  figures, 13 of which are in the appendices

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17270v1) [paper-pdf](http://arxiv.org/pdf/2502.17270v1)

**Authors**: Erwan Mahe, Sara Tucci-Piergiovanni

**Abstract**: Order fairness in distributed ledgers refers to properties that relate the order in which transactions are sent or received to the order in which they are eventually finalized, i.e., totally ordered. The study of such properties is relatively new and has been especially stimulated by the rise of Maximal Extractable Value (MEV) attacks in blockchain environments. Indeed, in many classical blockchain protocols, leaders are responsible for selecting the transactions to be included in blocks, which creates a clear vulnerability and opportunity for transaction order manipulation.   Unlike blockchains, DAG-based ledgers allow participants in the network to independently propose blocks, which are then arranged as vertices of a directed acyclic graph. Interestingly, leaders in DAG-based ledgers are elected only after the fact, once transactions are already part of the graph, to determine their total order. In other words, transactions are not chosen by single leaders; instead, they are collectively validated by the nodes, and leaders are only elected to establish an ordering. This approach intuitively reduces the risk of transaction manipulation and enhances fairness.   In this paper, we aim to quantify the capability of DAG-based ledgers to achieve order fairness. To this end, we define new variants of order fairness adapted to DAG-based ledgers and evaluate the impact of an adversary capable of compromising a limited number of nodes (below the one-third threshold) to reorder transactions. We analyze how often our order fairness properties are violated under different network conditions and parameterizations of the DAG algorithm, depending on the adversary's power.   Our study shows that DAG-based ledgers are still vulnerable to reordering attacks, as an adversary can coordinate a minority of Byzantine nodes to manipulate the DAG's structure.



## **16. REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective**

cs.LG

30 pages, 6 figures, 15 tables

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17254v1) [paper-pdf](http://arxiv.org/pdf/2502.17254v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Vincent Cohen-Addad, Johannes Gasteiger, Stephan Günnemann

**Abstract**: To circumvent the alignment of large language models (LLMs), current optimization-based adversarial attacks usually craft adversarial prompts by maximizing the likelihood of a so-called affirmative response. An affirmative response is a manually designed start of a harmful answer to an inappropriate request. While it is often easy to craft prompts that yield a substantial likelihood for the affirmative response, the attacked model frequently does not complete the response in a harmful manner. Moreover, the affirmative objective is usually not adapted to model-specific preferences and essentially ignores the fact that LLMs output a distribution over responses. If low attack success under such an objective is taken as a measure of robustness, the true robustness might be grossly overestimated. To alleviate these flaws, we propose an adaptive and semantic optimization problem over the population of responses. We derive a generally applicable objective via the REINFORCE policy-gradient formalism and demonstrate its efficacy with the state-of-the-art jailbreak algorithms Greedy Coordinate Gradient (GCG) and Projected Gradient Descent (PGD). For example, our objective doubles the attack success rate (ASR) on Llama3 and increases the ASR from 2% to 50% with circuit breaker defense.



## **17. Adversarial Training for Defense Against Label Poisoning Attacks**

cs.LG

Accepted at the International Conference on Learning Representations  (ICLR 2025)

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17121v1) [paper-pdf](http://arxiv.org/pdf/2502.17121v1)

**Authors**: Melis Ilayda Bal, Volkan Cevher, Michael Muehlebach

**Abstract**: As machine learning models grow in complexity and increasingly rely on publicly sourced data, such as the human-annotated labels used in training large language models, they become more vulnerable to label poisoning attacks. These attacks, in which adversaries subtly alter the labels within a training dataset, can severely degrade model performance, posing significant risks in critical applications. In this paper, we propose FLORAL, a novel adversarial training defense strategy based on support vector machines (SVMs) to counter these threats. Utilizing a bilevel optimization framework, we cast the training process as a non-zero-sum Stackelberg game between an attacker, who strategically poisons critical training labels, and the model, which seeks to recover from such attacks. Our approach accommodates various model architectures and employs a projected gradient descent algorithm with kernel SVMs for adversarial training. We provide a theoretical analysis of our algorithm's convergence properties and empirically evaluate FLORAL's effectiveness across diverse classification tasks. Compared to robust baselines and foundation models such as RoBERTa, FLORAL consistently achieves higher robust accuracy under increasing attacker budgets. These results underscore the potential of FLORAL to enhance the resilience of machine learning models against label poisoning threats, thereby ensuring robust classification in adversarial settings.



## **18. Improving the Transferability of Adversarial Examples by Inverse Knowledge Distillation**

cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17003v1) [paper-pdf](http://arxiv.org/pdf/2502.17003v1)

**Authors**: Wenyuan Wu, Zheng Liu, Yong Chen, Chao Su, Dezhong Peng, Xu Wang

**Abstract**: In recent years, the rapid development of deep neural networks has brought increased attention to the security and robustness of these models. While existing adversarial attack algorithms have demonstrated success in improving adversarial transferability, their performance remains suboptimal due to a lack of consideration for the discrepancies between target and source models. To address this limitation, we propose a novel method, Inverse Knowledge Distillation (IKD), designed to enhance adversarial transferability effectively. IKD introduces a distillation-inspired loss function that seamlessly integrates with gradient-based attack methods, promoting diversity in attack gradients and mitigating overfitting to specific model architectures. By diversifying gradients, IKD enables the generation of adversarial samples with superior generalization capabilities across different models, significantly enhancing their effectiveness in black-box attack scenarios. Extensive experiments on the ImageNet dataset validate the effectiveness of our approach, demonstrating substantial improvements in the transferability and attack success rates of adversarial samples across a wide range of models.



## **19. VGFL-SA: Vertical Graph Federated Learning Structure Attack Based on Contrastive Learning**

cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.16793v1) [paper-pdf](http://arxiv.org/pdf/2502.16793v1)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks (GNNs) have gained attention for their ability to learn representations from graph data. Due to privacy concerns and conflicts of interest that prevent clients from directly sharing graph data with one another, Vertical Graph Federated Learning (VGFL) frameworks have been developed. Recent studies have shown that VGFL is vulnerable to adversarial attacks that degrade performance. However, it is a common problem that client nodes are often unlabeled in the realm of VGFL. Consequently, the existing attacks, which rely on the availability of labeling information to obtain gradients, are inherently constrained in their applicability. This limitation precludes their deployment in practical, real-world environments. To address the above problems, we propose a novel graph adversarial attack against VGFL, referred to as VGFL-SA, to degrade the performance of VGFL by modifying the local clients structure without using labels. Specifically, VGFL-SA uses a contrastive learning method to complete the attack before the local clients are trained. VGFL-SA first accesses the graph structure and node feature information of the poisoned clients, and generates the contrastive views by node-degree-based edge augmentation and feature shuffling augmentation. Then, VGFL-SA uses the shared graph encoder to get the embedding of each view, and the gradients of the adjacency matrices are obtained by the contrastive function. Finally, perturbed edges are generated using gradient modification rules. We validated the performance of VGFL-SA by performing a node classification task on real-world datasets, and the results show that VGFL-SA achieves good attack effectiveness and transferability.



## **20. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

Accepted by ICLR2025

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2406.18849v4) [paper-pdf](http://arxiv.org/pdf/2406.18849v4)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released at https://github.com/Robin-WZQ/Dysca.



## **21. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16750v1) [paper-pdf](http://arxiv.org/pdf/2502.16750v1)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **22. Keeping up with dynamic attackers: Certifying robustness to adaptive online data poisoning**

cs.LG

Proceedings of the 28th International Conference on Artificial  Intelligence and Statistics (AISTATS) 2025, Mai Khao, Thailand. PMLR: Volume  258

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16737v1) [paper-pdf](http://arxiv.org/pdf/2502.16737v1)

**Authors**: Avinandan Bose, Laurent Lessard, Maryam Fazel, Krishnamurthy Dj Dvijotham

**Abstract**: The rise of foundation models fine-tuned on human feedback from potentially untrusted users has increased the risk of adversarial data poisoning, necessitating the study of robustness of learning algorithms against such attacks. Existing research on provable certified robustness against data poisoning attacks primarily focuses on certifying robustness for static adversaries who modify a fraction of the dataset used to train the model before the training algorithm is applied. In practice, particularly when learning from human feedback in an online sense, adversaries can observe and react to the learning process and inject poisoned samples that optimize adversarial objectives better than when they are restricted to poisoning a static dataset once, before the learning algorithm is applied. Indeed, it has been shown in prior work that online dynamic adversaries can be significantly more powerful than static ones. We present a novel framework for computing certified bounds on the impact of dynamic poisoning, and use these certificates to design robust learning algorithms. We give an illustration of the framework for the mean estimation and binary classification problems and outline directions for extending this in further work. The code to implement our certificates and replicate our results is available at https://github.com/Avinandan22/Certified-Robustness.



## **23. Towards Optimal Adversarial Robust Reinforcement Learning with Infinity Measurement Error**

cs.LG

arXiv admin note: substantial text overlap with arXiv:2402.02165

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16734v1) [paper-pdf](http://arxiv.org/pdf/2502.16734v1)

**Authors**: Haoran Li, Zicheng Zhang, Wang Luo, Congying Han, Jiayu Lv, Tiande Guo, Yudong Hu

**Abstract**: Ensuring the robustness of deep reinforcement learning (DRL) agents against adversarial attacks is critical for their trustworthy deployment. Recent research highlights the challenges of achieving state-adversarial robustness and suggests that an optimal robust policy (ORP) does not always exist, complicating the enforcement of strict robustness constraints. In this paper, we further explore the concept of ORP. We first introduce the Intrinsic State-adversarial Markov Decision Process (ISA-MDP), a novel formulation where adversaries cannot fundamentally alter the intrinsic nature of state observations. ISA-MDP, supported by empirical and theoretical evidence, universally characterizes decision-making under state-adversarial paradigms. We rigorously prove that within ISA-MDP, a deterministic and stationary ORP exists, aligning with the Bellman optimal policy. Our findings theoretically reveal that improving DRL robustness does not necessarily compromise performance in natural environments. Furthermore, we demonstrate the necessity of infinity measurement error (IME) in both $Q$-function and probability spaces to achieve ORP, unveiling vulnerabilities of previous DRL algorithms that rely on $1$-measurement errors. Motivated by these insights, we develop the Consistent Adversarial Robust Reinforcement Learning (CAR-RL) framework, which optimizes surrogates of IME. We apply CAR-RL to both value-based and policy-based DRL algorithms, achieving superior performance and validating our theoretical analysis.



## **24. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**

cs.LG

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.05374v2) [paper-pdf](http://arxiv.org/pdf/2502.05374v2)

**Authors**: Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu

**Abstract**: The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.



## **25. Uncovering the Hidden Threat of Text Watermarking from Users with Cross-Lingual Knowledge**

cs.CL

9 pages

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16699v1) [paper-pdf](http://arxiv.org/pdf/2502.16699v1)

**Authors**: Mansour Al Ghanim, Jiaqi Xue, Rochana Prih Hastuti, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: In this study, we delve into the hidden threats posed to text watermarking by users with cross-lingual knowledge. While most research focuses on watermarking methods for English, there is a significant gap in evaluating these methods in cross-lingual contexts. This oversight neglects critical adversary scenarios involving cross-lingual users, creating uncertainty regarding the effectiveness of cross-lingual watermarking. We assess four watermarking techniques across four linguistically rich languages, examining watermark resilience and text quality across various parameters and attacks. Our focus is on a realistic scenario featuring adversaries with cross-lingual expertise, evaluating the adequacy of current watermarking methods against such challenges.



## **26. AdverX-Ray: Ensuring X-Ray Integrity Through Frequency-Sensitive Adversarial VAEs**

cs.CV

SPIE Medical Imaging 2025 Runner-up 2025 Robert F. Wagner  All-Conference Best Student Paper Award

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16610v1) [paper-pdf](http://arxiv.org/pdf/2502.16610v1)

**Authors**: Francisco Caetano, Christiaan Viviers, Lena Filatova, Peter H. N. de With, Fons van der Sommen

**Abstract**: Ensuring the quality and integrity of medical images is crucial for maintaining diagnostic accuracy in deep learning-based Computer-Aided Diagnosis and Computer-Aided Detection (CAD) systems. Covariate shifts are subtle variations in the data distribution caused by different imaging devices or settings and can severely degrade model performance, similar to the effects of adversarial attacks. Therefore, it is vital to have a lightweight and fast method to assess the quality of these images prior to using CAD models. AdverX-Ray addresses this need by serving as an image-quality assessment layer, designed to detect covariate shifts effectively. This Adversarial Variational Autoencoder prioritizes the discriminator's role, using the suboptimal outputs of the generator as negative samples to fine-tune the discriminator's ability to identify high-frequency artifacts. Images generated by adversarial networks often exhibit severe high-frequency artifacts, guiding the discriminator to focus excessively on these components. This makes the discriminator ideal for this approach. Trained on patches from X-ray images of specific machine models, AdverX-Ray can evaluate whether a scan matches the training distribution, or if a scan from the same machine is captured under different settings. Extensive comparisons with various OOD detection methods show that AdverX-Ray significantly outperforms existing techniques, achieving a 96.2% average AUROC using only 64 random patches from an X-ray. Its lightweight and fast architecture makes it suitable for real-time applications, enhancing the reliability of medical imaging systems. The code and pretrained models are publicly available.



## **27. Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images**

cs.AI

Accepted to ICLR 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16593v1) [paper-pdf](http://arxiv.org/pdf/2502.16593v1)

**Authors**: Yubo Wang, Jianting Tang, Chaohu Liu, Linli Xu

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable image understanding and dialogue capabilities, allowing them to handle a variety of visual question answering tasks. However, their widespread availability raises concerns about unauthorized usage and copyright infringement, where users or individuals can develop their own LVLMs by fine-tuning published models. In this paper, we propose a novel method called Parameter Learning Attack (PLA) for tracking the copyright of LVLMs without modifying the original model. Specifically, we construct adversarial images through targeted attacks against the original model, enabling it to generate specific outputs. To ensure these attacks remain effective on potential fine-tuned models to trigger copyright tracking, we allow the original model to learn the trigger images by updating parameters in the opposite direction during the adversarial attack process. Notably, the proposed method can be applied after the release of the original model, thus not affecting the model's performance and behavior. To simulate real-world applications, we fine-tune the original model using various strategies across diverse datasets, creating a range of models for copyright verification. Extensive experiments demonstrate that our method can more effectively identify the original copyright of fine-tuned models compared to baseline methods. Therefore, this work provides a powerful tool for tracking copyrights and detecting unlicensed usage of LVLMs.



## **28. Robust Kernel Hypothesis Testing under Data Corruption**

stat.ML

22 pages, 2 figures, 2 algorithms

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2405.19912v2) [paper-pdf](http://arxiv.org/pdf/2405.19912v2)

**Authors**: Antonin Schrab, Ilmun Kim

**Abstract**: We propose a general method for constructing robust permutation tests under data corruption. The proposed tests effectively control the non-asymptotic type I error under data corruption, and we prove their consistency in power under minimal conditions. This contributes to the practical deployment of hypothesis tests for real-world applications with potential adversarial attacks. For the two-sample and independence settings, we show that our kernel robust tests are minimax optimal, in the sense that they are guaranteed to be non-asymptotically powerful against alternatives uniformly separated from the null in the kernel MMD and HSIC metrics at some optimal rate (tight with matching lower bound). We point out that existing differentially private tests can be adapted to be robust to data corruption, and we demonstrate in experiments that our proposed tests achieve much higher power than these private tests. Finally, we provide publicly available implementations and empirically illustrate the practicality of our robust tests.



## **29. Certified Causal Defense with Generalizable Robustness**

cs.LG

Submitted to AAAI

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2408.15451v2) [paper-pdf](http://arxiv.org/pdf/2408.15451v2)

**Authors**: Yiran Qiao, Yu Yin, Chen Chen, Jing Ma

**Abstract**: While machine learning models have proven effective across various scenarios, it is widely acknowledged that many models are vulnerable to adversarial attacks. Recently, there have emerged numerous efforts in adversarial defense. Among them, certified defense is well known for its theoretical guarantees against arbitrary adversarial perturbations on input within a certain range (e.g., $l_2$ ball). However, most existing works in this line struggle to generalize their certified robustness in other data domains with distribution shifts. This issue is rooted in the difficulty of eliminating the negative impact of spurious correlations on robustness in different domains. To address this problem, in this work, we propose a novel certified defense framework GLEAN, which incorporates a causal perspective into the generalization problem in certified defense. More specifically, our framework integrates a certifiable causal factor learning component to disentangle the causal relations and spurious correlations between input and label, and thereby exclude the negative effect of spurious correlations on defense. On top of that, we design a causally certified defense strategy to handle adversarial attacks on latent causal factors. In this way, our framework is not only robust against malicious noises on data in the training distribution but also can generalize its robustness across domains with distribution shifts. Extensive experiments on benchmark datasets validate the superiority of our framework in certified robustness generalization in different data domains. Code is available in the supplementary materials.



## **30. Unified Prompt Attack Against Text-to-Image Generation Models**

cs.CV

Accepted by IEEE T-PAMI 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16423v1) [paper-pdf](http://arxiv.org/pdf/2502.16423v1)

**Authors**: Duo Peng, Qiuhong Ke, Mark He Huang, Ping Hu, Jun Liu

**Abstract**: Text-to-Image (T2I) models have advanced significantly, but their growing popularity raises security concerns due to their potential to generate harmful images. To address these issues, we propose UPAM, a novel framework to evaluate the robustness of T2I models from an attack perspective. Unlike prior methods that focus solely on textual defenses, UPAM unifies the attack on both textual and visual defenses. Additionally, it enables gradient-based optimization, overcoming reliance on enumeration for improved efficiency and effectiveness. To handle cases where T2I models block image outputs due to defenses, we introduce Sphere-Probing Learning (SPL) to enable optimization even without image results. Following SPL, our model bypasses defenses, inducing the generation of harmful content. To ensure semantic alignment with attacker intent, we propose Semantic-Enhancing Learning (SEL) for precise semantic control. UPAM also prioritizes the naturalness of adversarial prompts using In-context Naturalness Enhancement (INE), making them harder for human examiners to detect. Additionally, we address the issue of iterative queries--common in prior methods and easily detectable by API defenders--by introducing Transferable Attack Learning (TAL), allowing effective attacks with minimal queries. Extensive experiments validate UPAM's superiority in effectiveness, efficiency, naturalness, and low query detection rates.



## **31. FedNIA: Noise-Induced Activation Analysis for Mitigating Data Poisoning in FL**

cs.LG

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16396v1) [paper-pdf](http://arxiv.org/pdf/2502.16396v1)

**Authors**: Ehsan Hallaji, Roozbeh Razavi-Far, Mehrdad Saif

**Abstract**: Federated learning systems are increasingly threatened by data poisoning attacks, where malicious clients compromise global models by contributing tampered updates. Existing defenses often rely on impractical assumptions, such as access to a central test dataset, or fail to generalize across diverse attack types, particularly those involving multiple malicious clients working collaboratively. To address this, we propose Federated Noise-Induced Activation Analysis (FedNIA), a novel defense framework to identify and exclude adversarial clients without relying on any central test dataset. FedNIA injects random noise inputs to analyze the layerwise activation patterns in client models leveraging an autoencoder that detects abnormal behaviors indicative of data poisoning. FedNIA can defend against diverse attack types, including sample poisoning, label flipping, and backdoors, even in scenarios with multiple attacking nodes. Experimental results on non-iid federated datasets demonstrate its effectiveness and robustness, underscoring its potential as a foundational approach for enhancing the security of federated learning systems.



## **32. A Framework for Evaluating Vision-Language Model Safety: Building Trust in AI for Public Sector Applications**

cs.CY

AAAI 2025 Workshop on AI for Social Impact: Bridging Innovations in  Finance, Social Media, and Crime Prevention

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16361v1) [paper-pdf](http://arxiv.org/pdf/2502.16361v1)

**Authors**: Maisha Binte Rashid, Pablo Rivas

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in public sector missions, necessitating robust evaluation of their safety and vulnerability to adversarial attacks. This paper introduces a novel framework to quantify adversarial risks in VLMs. We analyze model performance under Gaussian, salt-and-pepper, and uniform noise, identifying misclassification thresholds and deriving composite noise patches and saliency patterns that highlight vulnerable regions. These patterns are compared against the Fast Gradient Sign Method (FGSM) to assess their adversarial effectiveness. We propose a new Vulnerability Score that combines the impact of random noise and adversarial attacks, providing a comprehensive metric for evaluating model robustness.



## **33. Verification of Bit-Flip Attacks against Quantized Neural Networks**

cs.CR

37 pages, 13 figures, 14 tables

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16286v1) [paper-pdf](http://arxiv.org/pdf/2502.16286v1)

**Authors**: Yedi Zhang, Lei Huang, Pengfei Gao, Fu Song, Jun Sun, Jin Song Dong

**Abstract**: In the rapidly evolving landscape of neural network security, the resilience of neural networks against bit-flip attacks (i.e., an attacker maliciously flips an extremely small amount of bits within its parameter storage memory system to induce harmful behavior), has emerged as a relevant area of research. Existing studies suggest that quantization may serve as a viable defense against such attacks. Recognizing the documented susceptibility of real-valued neural networks to such attacks and the comparative robustness of quantized neural networks (QNNs), in this work, we introduce BFAVerifier, the first verification framework designed to formally verify the absence of bit-flip attacks or to identify all vulnerable parameters in a sound and rigorous manner. BFAVerifier comprises two integral components: an abstraction-based method and an MILP-based method. Specifically, we first conduct a reachability analysis with respect to symbolic parameters that represent the potential bit-flip attacks, based on a novel abstract domain with a sound guarantee. If the reachability analysis fails to prove the resilience of such attacks, then we encode this verification problem into an equivalent MILP problem which can be solved by off-the-shelf solvers. Therefore, BFAVerifier is sound, complete, and reasonably efficient. We conduct extensive experiments, which demonstrate its effectiveness and efficiency across various network architectures, quantization bit-widths, and adversary capabilities.



## **34. Your Diffusion Model is Secretly a Certifiably Robust Classifier**

cs.LG

Accepted by NeurIPS 2024. Also named as "Diffusion Models are  Certifiably Robust Classifiers"

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2402.02316v4) [paper-pdf](http://arxiv.org/pdf/2402.02316v4)

**Authors**: Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: Generative learning, recognized for its effective modeling of data distributions, offers inherent advantages in handling out-of-distribution instances, especially for enhancing robustness to adversarial attacks. Among these, diffusion classifiers, utilizing powerful diffusion models, have demonstrated superior empirical robustness. However, a comprehensive theoretical understanding of their robustness is still lacking, raising concerns about their vulnerability to stronger future attacks. In this study, we prove that diffusion classifiers possess $O(1)$ Lipschitzness, and establish their certified robustness, demonstrating their inherent resilience. To achieve non-constant Lipschitzness, thereby obtaining much tighter certified robustness, we generalize diffusion classifiers to classify Gaussian-corrupted data. This involves deriving the evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem. Experimental results show the superior certified robustness of these Noised Diffusion Classifiers (NDCs). Notably, we achieve over 80% and 70% certified robustness on CIFAR-10 under adversarial perturbations with \(\ell_2\) norms less than 0.25 and 0.5, respectively, using a single off-the-shelf diffusion model without any additional data.



## **35. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

cs.CL

Our study requires further in-depth research to ensure the  comprehensiveness and adequacy of the methodology

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2412.12145v4) [paper-pdf](http://arxiv.org/pdf/2412.12145v4)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}



## **36. A Survey of Model Extraction Attacks and Defenses in Distributed Computing Environments**

cs.CR

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16065v1) [paper-pdf](http://arxiv.org/pdf/2502.16065v1)

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong

**Abstract**: Model Extraction Attacks (MEAs) threaten modern machine learning systems by enabling adversaries to steal models, exposing intellectual property and training data. With the increasing deployment of machine learning models in distributed computing environments, including cloud, edge, and federated learning settings, each paradigm introduces distinct vulnerabilities and challenges. Without a unified perspective on MEAs across these distributed environments, organizations risk fragmented defenses, inadequate risk assessments, and substantial economic and privacy losses. This survey is motivated by the urgent need to understand how the unique characteristics of cloud, edge, and federated deployments shape attack vectors and defense requirements. We systematically examine the evolution of attack methodologies and defense mechanisms across these environments, demonstrating how environmental factors influence security strategies in critical sectors such as autonomous vehicles, healthcare, and financial services. By synthesizing recent advances in MEAs research and discussing the limitations of current evaluation practices, this survey provides essential insights for developing robust and adaptive defense strategies. Our comprehensive approach highlights the importance of integrating protective measures across the entire distributed computing landscape to ensure the secure deployment of machine learning models.



## **37. Human-AI Collaboration in Cloud Security: Cognitive Hierarchy-Driven Deep Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16054v1) [paper-pdf](http://arxiv.org/pdf/2502.16054v1)

**Authors**: Zahra Aref, Sheng Wei, Narayan B. Mandayam

**Abstract**: Given the complexity of multi-tenant cloud environments and the need for real-time threat mitigation, Security Operations Centers (SOCs) must integrate AI-driven adaptive defenses against Advanced Persistent Threats (APTs). However, SOC analysts struggle with countering adaptive adversarial tactics, necessitating intelligent decision-support frameworks. To enhance human-AI collaboration in SOCs, we propose a Cognitive Hierarchy Theory-driven Deep Q-Network (CHT-DQN) framework that models SOC analysts' decision-making against AI-driven APT bots. The SOC analyst (defender) operates at cognitive level-1, anticipating attacker strategies, while the APT bot (attacker) follows a level-0 exploitative policy. By incorporating CHT into DQN, our framework enhances SOC defense strategies via Attack Graph (AG)-based reinforcement learning. Simulation experiments across varying AG complexities show that CHT-DQN achieves higher data protection and lower action discrepancies compared to standard DQN. A theoretical lower bound analysis further validates its superior Q-value performance. A human-in-the-loop (HITL) evaluation on Amazon Mechanical Turk (MTurk) reveals that SOC analysts using CHT-DQN-driven transition probabilities align better with adaptive attackers, improving data protection. Additionally, human decision patterns exhibit risk aversion after failure and risk-seeking behavior after success, aligning with Prospect Theory. These findings underscore the potential of integrating cognitive modeling into deep reinforcement learning to enhance SOC operations and develop real-time adaptive cloud security mechanisms.



## **38. A Multi-Scale Isolation Forest Approach for Real-Time Detection and Filtering of FGSM Adversarial Attacks in Video Streams of Autonomous Vehicles**

cs.CV

17 pages, 7 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16044v1) [paper-pdf](http://arxiv.org/pdf/2502.16044v1)

**Authors**: Richard Abhulimhen, Negash Begashaw, Gurcan Comert, Chunheng Zhao, Pierluigi Pisu

**Abstract**: Deep Neural Networks (DNNs) have demonstrated remarkable success across a wide range of tasks, particularly in fields such as image classification. However, DNNs are highly susceptible to adversarial attacks, where subtle perturbations are introduced to input images, leading to erroneous model outputs. In today's digital era, ensuring the security and integrity of images processed by DNNs is of critical importance. One of the most prominent adversarial attack methods is the Fast Gradient Sign Method (FGSM), which perturbs images in the direction of the loss gradient to deceive the model.   This paper presents a novel approach for detecting and filtering FGSM adversarial attacks in image processing tasks. Our proposed method evaluates 10,000 images, each subjected to five different levels of perturbation, characterized by $\epsilon$ values of 0.01, 0.02, 0.05, 0.1, and 0.2. These perturbations are applied in the direction of the loss gradient. We demonstrate that our approach effectively filters adversarially perturbed images, mitigating the impact of FGSM attacks.   The method is implemented in Python, and the source code is publicly available on GitHub for reproducibility and further research.



## **39. Overcoming Intensity Limits for Long-Distance Quantum Key Distribution**

quant-ph

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2412.20265v3) [paper-pdf](http://arxiv.org/pdf/2412.20265v3)

**Authors**: Ibrahim Almosallam

**Abstract**: Quantum Key Distribution (QKD) enables the sharing of cryptographic keys secured by quantum mechanics. The BB84 protocol assumed single-photon sources, but practical systems rely on weak coherent pulses vulnerable to photon-number-splitting (PNS) attacks. The Gottesman-Lo-L\"utkenhaus-Preskill (GLLP) framework addressed these imperfections, deriving secure key rate bounds under limited PNS scenarios. The Decoy-state protocol further improved performance by refining single-photon yield estimates, but still considered multi-photon states as insecure, thereby limiting intensities and constraining key rate and distance. More recently, finite-key security bounds for decoy-state QKD have been extended to address general attacks, ensuring security against adversaries capable of exploiting arbitrary strategies. In this work, we focus on a specific class of attacks, the generalized PNS attack, and demonstrate that higher pulse intensities can be securely used by employing Bayesian inference to estimate key parameters directly from observed data. By raising the pulse intensity to 10 photons, we achieve a 50-fold increase in key rate and a 62.2% increase in operational range (about 200 km) compared to the decoy-state protocol. Furthermore, we accurately model after-pulsing using a Hidden Markov Model and reveal inaccuracies in decoy-state calculations that may produce erroneous key-rate estimates. While this methodology does not address all possible attacks, it provides a new approach to security proofs in QKD by shifting from worst-case assumption analysis to observation-dependent inference, advancing the reach and efficiency of discrete-variable QKD protocols.



## **40. Cross-Model Transferability of Adversarial Patches in Real-time Segmentation for Autonomous Driving**

cs.CV

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16012v1) [paper-pdf](http://arxiv.org/pdf/2502.16012v1)

**Authors**: Prashant Shekhar, Bidur Devkota, Dumindu Samaraweera, Laxima Niure Kandel, Manoj Babu

**Abstract**: Adversarial attacks pose a significant threat to deep learning models, particularly in safety-critical applications like healthcare and autonomous driving. Recently, patch based attacks have demonstrated effectiveness in real-time inference scenarios owing to their 'drag and drop' nature. Following this idea for Semantic Segmentation (SS), here we propose a novel Expectation Over Transformation (EOT) based adversarial patch attack that is more realistic for autonomous vehicles. To effectively train this attack we also propose a 'simplified' loss function that is easy to analyze and implement. Using this attack as our basis, we investigate whether adversarial patches once optimized on a specific SS model, can fool other models or architectures. We conduct a comprehensive cross-model transferability analysis of adversarial patches trained on SOTA Convolutional Neural Network (CNN) models such PIDNet-S, PIDNet-M and PIDNet-L, among others. Additionally, we also include the Segformer model to study transferability to Vision Transformers (ViTs). All of our analysis is conducted on the widely used Cityscapes dataset. Our study reveals key insights into how model architectures (CNN vs CNN or CNN vs. Transformer-based) influence attack susceptibility. In particular, we conclude that although the transferability (effectiveness) of attacks on unseen images of any dimension is really high, the attacks trained against one particular model are minimally effective on other models. And this was found to be true for both ViT and CNN based models. Additionally our results also indicate that for CNN-based models, the repercussions of patch attacks are local, unlike ViTs. Per-class analysis reveals that simple-classes like 'sky' suffer less misclassification than others. The code for the project is available at: https://github.com/p-shekhar/adversarial-patch-transferability



## **41. A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

cs.CV

21 pages, 7 figures, 10 tables

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2408.10901v3) [paper-pdf](http://arxiv.org/pdf/2408.10901v3)

**Authors**: Zhongliang Guo, Chun Tong Lei, Lei Fang, Shuai Zhao, Yifei Qian, Jingyu Lin, Zeyu Wang, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in generative AI, particularly Latent Diffusion Models (LDMs), have revolutionized image synthesis and manipulation. However, these generative techniques raises concerns about data misappropriation and intellectual property infringement. Adversarial attacks on machine learning models have been extensively studied, and a well-established body of research has extended these techniques as a benign metric to prevent the underlying misuse of generative AI. Current approaches to safeguarding images from manipulation by LDMs are limited by their reliance on model-specific knowledge and their inability to significantly degrade semantic quality of generated images. In response to these shortcomings, we propose the Posterior Collapse Attack (PCA) based on the observation that VAEs suffer from posterior collapse during training. Our method minimizes dependence on the white-box information of target models to get rid of the implicit reliance on model-specific knowledge. By accessing merely a small amount of LDM parameters, in specific merely the VAE encoder of LDMs, our method causes a substantial semantic collapse in generation quality, particularly in perceptual consistency, and demonstrates strong transferability across various model architectures. Experimental results show that PCA achieves superior perturbation effects on image generation of LDMs with lower runtime and VRAM. Our method outperforms existing techniques, offering a more robust and generalizable solution that is helpful in alleviating the socio-technical challenges posed by the rapidly evolving landscape of generative AI.



## **42. Defending Jailbreak Prompts via In-Context Adversarial Game**

cs.LG

EMNLP 2024 Main Paper

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2402.13148v3) [paper-pdf](http://arxiv.org/pdf/2402.13148v3)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.



## **43. A First Principles Approach to Trust-Based Recommendation Systems in Social Networks**

cs.IR

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2407.00062v2) [paper-pdf](http://arxiv.org/pdf/2407.00062v2)

**Authors**: Paras Stefanopoulos, Sourin Chatterjee, Ahad N. Zehmakan

**Abstract**: This paper explores recommender systems in social networks which leverage information such as item rating, intra-item similarities, and trust graph. We demonstrate that item-rating information is more influential than other information types in a collaborative filtering approach. The trust graph-based approaches were found to be more robust to network adversarial attacks due to hard-to-manipulate trust structures. Intra-item information, although sub-optimal in isolation, enhances the consistency of predictions and lower-end performance when fused with other information forms. Additionally, the Weighted Average framework is introduced, enabling the construction of recommendation systems around any user-to-user similarity metric. All the codes are publicly available on GitHub.



## **44. A Comprehensive Survey on the Trustworthiness of Large Language Models in Healthcare**

cs.CY

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15871v1) [paper-pdf](http://arxiv.org/pdf/2502.15871v1)

**Authors**: Manar Aljohani, Jun Hou, Sindhura Kommu, Xuan Wang

**Abstract**: The application of large language models (LLMs) in healthcare has the potential to revolutionize clinical decision-making, medical research, and patient care. As LLMs are increasingly integrated into healthcare systems, several critical challenges must be addressed to ensure their reliable and ethical deployment. These challenges include truthfulness, where models generate misleading information; privacy, with risks of unintentional data retention; robustness, requiring defenses against adversarial attacks; fairness, addressing biases in clinical outcomes; explainability, ensuring transparent decision-making; and safety, mitigating risks of misinformation and medical errors. Recently, researchers have begun developing benchmarks and evaluation frameworks to systematically assess the trustworthiness of LLMs. However, the trustworthiness of LLMs in healthcare remains underexplored, lacking a systematic review that provides a comprehensive understanding and future insights into this area. This survey bridges this gap by providing a comprehensive overview of the recent research of existing methodologies and solutions aimed at mitigating the above risks in healthcare. By focusing on key trustworthiness dimensions including truthfulness, privacy and safety, robustness, fairness and bias, and explainability, we present a thorough analysis of how these issues impact the reliability and ethical use of LLMs in healthcare. This paper highlights ongoing efforts and offers insights into future research directions to ensure the safe and trustworthy deployment of LLMs in healthcare.



## **45. Model Privacy: A Unified Framework to Understand Model Stealing Attacks and Defenses**

cs.LG

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15567v1) [paper-pdf](http://arxiv.org/pdf/2502.15567v1)

**Authors**: Ganghua Wang, Yuhong Yang, Jie Ding

**Abstract**: The use of machine learning (ML) has become increasingly prevalent in various domains, highlighting the importance of understanding and ensuring its safety. One pressing concern is the vulnerability of ML applications to model stealing attacks. These attacks involve adversaries attempting to recover a learned model through limited query-response interactions, such as those found in cloud-based services or on-chip artificial intelligence interfaces. While existing literature proposes various attack and defense strategies, these often lack a theoretical foundation and standardized evaluation criteria. In response, this work presents a framework called ``Model Privacy'', providing a foundation for comprehensively analyzing model stealing attacks and defenses. We establish a rigorous formulation for the threat model and objectives, propose methods to quantify the goodness of attack and defense strategies, and analyze the fundamental tradeoffs between utility and privacy in ML models. Our developed theory offers valuable insights into enhancing the security of ML models, especially highlighting the importance of the attack-specific structure of perturbations for effective defenses. We demonstrate the application of model privacy from the defender's perspective through various learning scenarios. Extensive experiments corroborate the insights and the effectiveness of defense mechanisms developed under the proposed framework.



## **46. A Defensive Framework Against Adversarial Attacks on Machine Learning-Based Network Intrusion Detection Systems**

cs.CR

Accepted to IEEE AI+ TrustCom 2024

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15561v1) [paper-pdf](http://arxiv.org/pdf/2502.15561v1)

**Authors**: Benyamin Tafreshian, Shengzhi Zhang

**Abstract**: As cyberattacks become increasingly sophisticated, advanced Network Intrusion Detection Systems (NIDS) are critical for modern network security. Traditional signature-based NIDS are inadequate against zero-day and evolving attacks. In response, machine learning (ML)-based NIDS have emerged as promising solutions; however, they are vulnerable to adversarial evasion attacks that subtly manipulate network traffic to bypass detection. To address this vulnerability, we propose a novel defensive framework that enhances the robustness of ML-based NIDS by simultaneously integrating adversarial training, dataset balancing techniques, advanced feature engineering, ensemble learning, and extensive model fine-tuning. We validate our framework using the NSL-KDD and UNSW-NB15 datasets. Experimental results show, on average, a 35% increase in detection accuracy and a 12.5% reduction in false positives compared to baseline models, particularly under adversarial conditions. The proposed defense against adversarial attacks significantly advances the practical deployment of robust ML-based NIDS in real-world networks.



## **47. Single-pass Detection of Jailbreaking Input in Large Language Models**

cs.LG

Accepted in TMLR 2025

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15435v1) [paper-pdf](http://arxiv.org/pdf/2502.15435v1)

**Authors**: Leyla Naz Candogan, Yongtao Wu, Elias Abad Rocamora, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Defending aligned Large Language Models (LLMs) against jailbreaking attacks is a challenging problem, with existing approaches requiring multiple requests or even queries to auxiliary LLMs, making them computationally heavy. Instead, we focus on detecting jailbreaking input in a single forward pass. Our method, called Single Pass Detection SPD, leverages the information carried by the logits to predict whether the output sentence will be harmful. This allows us to defend in just one forward pass. SPD can not only detect attacks effectively on open-source models, but also minimizes the misclassification of harmless inputs. Furthermore, we show that SPD remains effective even without complete logit access in GPT-3.5 and GPT-4. We believe that our proposed method offers a promising approach to efficiently safeguard LLMs against adversarial attacks.



## **48. Adversarial Prompt Evaluation: Systematic Benchmarking of Guardrails Against Prompt Input Attacks on LLMs**

cs.CR

NeurIPS 2024, Safe Generative AI Workshop

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15427v1) [paper-pdf](http://arxiv.org/pdf/2502.15427v1)

**Authors**: Giulio Zizzo, Giandomenico Cornacchia, Kieran Fraser, Muhammad Zaid Hameed, Ambrish Rawat, Beat Buesser, Mark Purcell, Pin-Yu Chen, Prasanna Sattigeri, Kush Varshney

**Abstract**: As large language models (LLMs) become integrated into everyday applications, ensuring their robustness and security is increasingly critical. In particular, LLMs can be manipulated into unsafe behaviour by prompts known as jailbreaks. The variety of jailbreak styles is growing, necessitating the use of external defences known as guardrails. While many jailbreak defences have been proposed, not all defences are able to handle new out-of-distribution attacks due to the narrow segment of jailbreaks used to align them. Moreover, the lack of systematisation around defences has created significant gaps in their practical application. In this work, we perform systematic benchmarking across 15 different defences, considering a broad swathe of malicious and benign datasets. We find that there is significant performance variation depending on the style of jailbreak a defence is subject to. Additionally, we show that based on current datasets available for evaluation, simple baselines can display competitive out-of-distribution performance compared to many state-of-the-art defences. Code is available at https://github.com/IBM/Adversarial-Prompt-Evaluation.



## **49. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

eess.SP

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2411.18220v3) [paper-pdf](http://arxiv.org/pdf/2411.18220v3)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.



## **50. Gröbner Basis Cryptanalysis of Ciminion and Hydra**

cs.CR

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2405.05040v4) [paper-pdf](http://arxiv.org/pdf/2405.05040v4)

**Authors**: Matthias Johann Steiner

**Abstract**: Ciminion and Hydra are two recently introduced symmetric key Pseudo-Random Functions for Multi-Party Computation applications. For efficiency, both primitives utilize quadratic permutations at round level. Therefore, polynomial system solving-based attacks pose a serious threat to these primitives. For Ciminion, we construct a quadratic degree reverse lexicographic (DRL) Gr\"obner basis for the iterated polynomial model via linear transformations. With the Gr\"obner basis we can simplify cryptanalysis, as we no longer need to impose genericity assumptions to derive complexity estimates. For Hydra, with the help of a computer algebra program like SageMath we construct a DRL Gr\"obner basis for the iterated model via linear transformations and a linear change of coordinates. In the Hydra proposal it was claimed that $r_\mathcal{H} = 31$ rounds are sufficient to provide $128$ bits of security against Gr\"obner basis attacks for an ideal adversary with $\omega = 2$. However, via our Hydra Gr\"obner basis standard term order conversion to a lexicographic (LEX) Gr\"obner basis requires just $126$ bits with $\omega = 2$. Moreover, using a dedicated polynomial system solving technique up to $r_\mathcal{H} = 33$ rounds can be attacked below $128$ bits for an ideal adversary.



