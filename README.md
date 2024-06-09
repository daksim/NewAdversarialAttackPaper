# Latest Adversarial Attack Papers
**update at 2024-06-09 20:17:43**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Improving Alignment and Robustness with Short Circuiting**

cs.LG

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04313v1) [paper-pdf](http://arxiv.org/pdf/2406.04313v1)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that "short-circuits" models as they respond with harmful outputs. Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, short-circuiting directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, short-circuiting allows the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.



## **2. NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label Noise**

cs.LG

Submitted to the 38th Conference on Neural Information Processing  Systems (NeurIPS 2024) Track on Datasets and Benchmarks

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04299v1) [paper-pdf](http://arxiv.org/pdf/2406.04299v1)

**Authors**: Zhonghao Wang, Danyu Sun, Sheng Zhou, Haobo Wang, Jiapei Fan, Longtao Huang, Jiajun Bu

**Abstract**: Graph Neural Networks (GNNs) exhibit strong potential in node classification task through a message-passing mechanism. However, their performance often hinges on high-quality node labels, which are challenging to obtain in real-world scenarios due to unreliable sources or adversarial attacks. Consequently, label noise is common in real-world graph data, negatively impacting GNNs by propagating incorrect information during training. To address this issue, the study of Graph Neural Networks under Label Noise (GLN) has recently gained traction. However, due to variations in dataset selection, data splitting, and preprocessing techniques, the community currently lacks a comprehensive benchmark, which impedes deeper understanding and further development of GLN. To fill this gap, we introduce NoisyGL in this paper, the first comprehensive benchmark for graph neural networks under label noise. NoisyGL enables fair comparisons and detailed analyses of GLN methods on noisy labeled graph data across various datasets, with unified experimental settings and interface. Our benchmark has uncovered several important insights that were missed in previous research, and we believe these findings will be highly beneficial for future studies. We hope our open-source benchmark library will foster further advancements in this field. The code of the benchmark can be found in https://github.com/eaglelab-zju/NoisyGL.



## **3. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models**

cs.CR

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.00083v2) [paper-pdf](http://arxiv.org/pdf/2406.00083v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, Qian Lou

**Abstract**: Large Language Models (LLMs) are constrained by outdated information and a tendency to generate incorrect data, commonly referred to as "hallucinations." Retrieval-Augmented Generation (RAG) addresses these limitations by combining the strengths of retrieval-based methods and generative models. This approach involves retrieving relevant information from a large, up-to-date dataset and using it to enhance the generation process, leading to more accurate and contextually appropriate responses. Despite its benefits, RAG introduces a new attack surface for LLMs, particularly because RAG databases are often sourced from public data, such as the web. In this paper, we propose \TrojRAG{} to identify the vulnerabilities and attacks on retrieval parts (RAG database) and their indirect attacks on generative parts (LLMs). Specifically, we identify that poisoning several customized content passages could achieve a retrieval backdoor, where the retrieval works well for clean queries but always returns customized poisoned adversarial queries. Triggers and poisoned passages can be highly customized to implement various attacks. For example, a trigger could be a semantic group like "The Republican Party, Donald Trump, etc." Adversarial passages can be tailored to different contents, not only linked to the triggers but also used to indirectly attack generative LLMs without modifying them. These attacks can include denial-of-service attacks on RAG and semantic steering attacks on LLM generations conditioned by the triggers. Our experiments demonstrate that by just poisoning 10 adversarial passages can induce 98.2\% success rate to retrieve the adversarial passages. Then, these passages can increase the reject ratio of RAG-based GPT-4 from 0.01\% to 74.6\% or increase the rate of negative responses from 0.22\% to 72\% for targeted queries.



## **4. Batch-in-Batch: a new adversarial training framework for initial perturbation and sample selection**

cs.LG

29 pages, 11 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04070v1) [paper-pdf](http://arxiv.org/pdf/2406.04070v1)

**Authors**: Yinting Wu, Pai Peng, Bo Cai, Le Li, .

**Abstract**: Adversarial training methods commonly generate independent initial perturbation for adversarial samples from a simple uniform distribution, and obtain the training batch for the classifier without selection. In this work, we propose a simple yet effective training framework called Batch-in-Batch (BB) to enhance models robustness. It involves specifically a joint construction of initial values that could simultaneously generates $m$ sets of perturbations from the original batch set to provide more diversity for adversarial samples; and also includes various sample selection strategies that enable the trained models to have smoother losses and avoid overconfident outputs. Through extensive experiments on three benchmark datasets (CIFAR-10, SVHN, CIFAR-100) with two networks (PreActResNet18 and WideResNet28-10) that are used in both the single-step (Noise-Fast Gradient Sign Method, N-FGSM) and multi-step (Projected Gradient Descent, PGD-10) adversarial training, we show that models trained within the BB framework consistently have higher adversarial accuracy across various adversarial settings, notably achieving over a 13% improvement on the SVHN dataset with an attack radius of 8/255 compared to the N-FGSM baseline model. Furthermore, experimental analysis of the efficiency of both the proposed initial perturbation method and sample selection strategies validates our insights. Finally, we show that our framework is cost-effective in terms of computational resources, even with a relatively large value of $m$.



## **5. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

cs.CV

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04031v1) [paper-pdf](http://arxiv.org/pdf/2406.04031v1)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.



## **6. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

cs.CL

Competition Report

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.14461v2) [paper-pdf](http://arxiv.org/pdf/2404.14461v2)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.



## **7. Verifiably Robust Conformal Prediction**

cs.LO

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2405.18942v2) [paper-pdf](http://arxiv.org/pdf/2405.18942v2)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces VRCP (Verifiably Robust Conformal Prediction), a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.



## **8. Behavior-Targeted Attack on Reinforcement Learning with Limited Access to Victim's Policy**

cs.LG

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03862v1) [paper-pdf](http://arxiv.org/pdf/2406.03862v1)

**Authors**: Shojiro Yamabe, Kazuto Fukuchi, Ryoma Senda, Jun Sakuma

**Abstract**: This study considers the attack on reinforcement learning agents where the adversary aims to control the victim's behavior as specified by the adversary by adding adversarial modifications to the victim's state observation. While some attack methods reported success in manipulating the victim agent's behavior, these methods often rely on environment-specific heuristics. In addition, all existing attack methods require white-box access to the victim's policy. In this study, we propose a novel method for manipulating the victim agent in the black-box (i.e., the adversary is allowed to observe the victim's state and action only) and no-box (i.e., the adversary is allowed to observe the victim's state only) setting without requiring environment-specific heuristics. Our attack method is formulated as a bi-level optimization problem that is reduced to a distribution matching problem and can be solved by an existing imitation learning algorithm in the black-box and no-box settings. Empirical evaluations on several reinforcement learning benchmarks show that our proposed method has superior attack performance to baselines.



## **9. Exploiting Global Graph Homophily for Generalized Defense in Graph Neural Networks**

cs.LG

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03833v1) [paper-pdf](http://arxiv.org/pdf/2406.03833v1)

**Authors**: Duanyu Li, Huijun Wu, Min Xie, Xugang Wu, Zhenwei Wu, Wenzhe Zhang

**Abstract**: Graph neural network (GNN) models play a pivotal role in numerous tasks involving graph-related data analysis. Despite their efficacy, similar to other deep learning models, GNNs are susceptible to adversarial attacks. Even minor perturbations in graph data can induce substantial alterations in model predictions. While existing research has explored various adversarial defense techniques for GNNs, the challenge of defending against adversarial attacks on real-world scale graph data remains largely unresolved. On one hand, methods reliant on graph purification and preprocessing tend to excessively emphasize local graph information, leading to sub-optimal defensive outcomes. On the other hand, approaches rooted in graph structure learning entail significant time overheads, rendering them impractical for large-scale graphs. In this paper, we propose a new defense method named Talos, which enhances the global, rather than local, homophily of graphs as a defense. Experiments show that the proposed approach notably outperforms state-of-the-art defense approaches, while imposing little computational overhead.



## **10. AutoJailbreak: Exploring Jailbreak Attacks and Defenses through a Dependency Lens**

cs.CR

32 pages, 2 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03805v1) [paper-pdf](http://arxiv.org/pdf/2406.03805v1)

**Authors**: Lin Lu, Hai Yan, Zenghui Yuan, Jiawen Shi, Wenqi Wei, Pin-Yu Chen, Pan Zhou

**Abstract**: Jailbreak attacks in large language models (LLMs) entail inducing the models to generate content that breaches ethical and legal norm through the use of malicious prompts, posing a substantial threat to LLM security. Current strategies for jailbreak attack and defense often focus on optimizing locally within specific algorithmic frameworks, resulting in ineffective optimization and limited scalability. In this paper, we present a systematic analysis of the dependency relationships in jailbreak attack and defense techniques, generalizing them to all possible attack surfaces. We employ directed acyclic graphs (DAGs) to position and analyze existing jailbreak attacks, defenses, and evaluation methodologies, and propose three comprehensive, automated, and logical frameworks. \texttt{AutoAttack} investigates dependencies in two lines of jailbreak optimization strategies: genetic algorithm (GA)-based attacks and adversarial-generation-based attacks, respectively. We then introduce an ensemble jailbreak attack to exploit these dependencies. \texttt{AutoDefense} offers a mixture-of-defenders approach by leveraging the dependency relationships in pre-generative and post-generative defense strategies. \texttt{AutoEvaluation} introduces a novel evaluation method that distinguishes hallucinations, which are often overlooked, from jailbreak attack and defense responses. Through extensive experiments, we demonstrate that the proposed ensemble jailbreak attack and defense framework significantly outperforms existing research.



## **11. Transferable Availability Poisoning Attacks**

cs.CR

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2310.05141v2) [paper-pdf](http://arxiv.org/pdf/2310.05141v2)

**Authors**: Yiyong Liu, Michael Backes, Xiao Zhang

**Abstract**: We consider availability data poisoning attacks, where an adversary aims to degrade the overall test accuracy of a machine learning model by crafting small perturbations to its training data. Existing poisoning strategies can achieve the attack goal but assume the victim to employ the same learning method as what the adversary uses to mount the attack. In this paper, we argue that this assumption is strong, since the victim may choose any learning algorithm to train the model as long as it can achieve some targeted performance on clean data. Empirically, we observe a large decrease in the effectiveness of prior poisoning attacks if the victim employs an alternative learning algorithm. To enhance the attack transferability, we propose Transferable Poisoning, which first leverages the intrinsic characteristics of alignment and uniformity to enable better unlearnability within contrastive learning, and then iteratively utilizes the gradient information from supervised and unsupervised contrastive learning paradigms to generate the poisoning perturbations. Through extensive experiments on image benchmarks, we show that our transferable poisoning attack can produce poisoned samples with significantly improved transferability, not only applicable to the two learners used to devise the attack but also to learning algorithms and even paradigms beyond.



## **12. Backdoor Attack with Sparse and Invisible Trigger**

cs.CV

This paper was accepted by IEEE Transactions on Information Forensics  and Security (TIFS). The first two authors contributed equally to this work.  14 pages

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2306.06209v3) [paper-pdf](http://arxiv.org/pdf/2306.06209v3)

**Authors**: Yinghua Gao, Yiming Li, Xueluan Gong, Zhifeng Li, Shu-Tao Xia, Qian Wang

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where the adversary manipulates a small portion of training data such that the victim model predicts normally on the benign samples but classifies the triggered samples as the target class. The backdoor attack is an emerging yet threatening training-phase threat, leading to serious risks in DNN-based applications. In this paper, we revisit the trigger patterns of existing backdoor attacks. We reveal that they are either visible or not sparse and therefore are not stealthy enough. More importantly, it is not feasible to simply combine existing methods to design an effective sparse and invisible backdoor attack. To address this problem, we formulate the trigger generation as a bi-level optimization problem with sparsity and invisibility constraints and propose an effective method to solve it. The proposed method is dubbed sparse and invisible backdoor attack (SIBA). We conduct extensive experiments on benchmark datasets under different settings, which verify the effectiveness of our attack and its resistance to existing backdoor defenses. The codes for reproducing main experiments are available at \url{https://github.com/YinghuaGao/SIBA}.



## **13. Principles of Designing Robust Remote Face Anti-Spoofing Systems**

cs.CV

Under review

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03684v1) [paper-pdf](http://arxiv.org/pdf/2406.03684v1)

**Authors**: Xiang Xu, Tianchen Zhao, Zheng Zhang, Zhihua Li, Jon Wu, Alessandro Achille, Mani Srivastava

**Abstract**: Protecting digital identities of human face from various attack vectors is paramount, and face anti-spoofing plays a crucial role in this endeavor. Current approaches primarily focus on detecting spoofing attempts within individual frames to detect presentation attacks. However, the emergence of hyper-realistic generative models capable of real-time operation has heightened the risk of digitally generated attacks. In light of these evolving threats, this paper aims to address two key aspects. First, it sheds light on the vulnerabilities of state-of-the-art face anti-spoofing methods against digital attacks. Second, it presents a comprehensive taxonomy of common threats encountered in face anti-spoofing systems. Through a series of experiments, we demonstrate the limitations of current face anti-spoofing detection techniques and their failure to generalize to novel digital attack scenarios. Notably, the existing models struggle with digital injection attacks including adversarial noise, realistic deepfake attacks, and digital replay attacks. To aid in the design and implementation of robust face anti-spoofing systems resilient to these emerging vulnerabilities, the paper proposes key design principles from model accuracy and robustness to pipeline robustness and even platform robustness. Especially, we suggest to implement the proactive face anti-spoofing system using active sensors to significant reduce the risks for unseen attack vectors and improve the user experience.



## **14. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2401.17263v3) [paper-pdf](http://arxiv.org/pdf/2401.17263v3)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo



## **15. Ranking Manipulation for Conversational Search Engines**

cs.CL

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03589v1) [paper-pdf](http://arxiv.org/pdf/2406.03589v1)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.



## **16. Distributional Adversarial Loss**

cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03458v1) [paper-pdf](http://arxiv.org/pdf/2406.03458v1)

**Authors**: Saba Ahmadi, Siddharth Bhandari, Avrim Blum, Chen Dan, Prabhav Jain

**Abstract**: A major challenge in defending against adversarial attacks is the enormous space of possible attacks that even a simple adversary might perform. To address this, prior work has proposed a variety of defenses that effectively reduce the size of this space. These include randomized smoothing methods that add noise to the input to take away some of the adversary's impact. Another approach is input discretization which limits the adversary's possible number of actions.   Motivated by these two approaches, we introduce a new notion of adversarial loss which we call distributional adversarial loss, to unify these two forms of effectively weakening an adversary. In this notion, we assume for each original example, the allowed adversarial perturbation set is a family of distributions (e.g., induced by a smoothing procedure), and the adversarial loss over each example is the maximum loss over all the associated distributions. The goal is to minimize the overall adversarial loss.   We show generalization guarantees for our notion of adversarial loss in terms of the VC-dimension of the hypothesis class and the size of the set of allowed adversarial distributions associated with each input. We also investigate the role of randomness in achieving robustness against adversarial attacks in the methods described above. We show a general derandomization technique that preserves the extent of a randomized classifier's robustness against adversarial attacks. We corroborate the procedure experimentally via derandomizing the Random Projection Filters framework of \cite{dong2023adversarial}. Our procedure also improves the robustness of the model against various adversarial attacks.



## **17. CR-UTP: Certified Robustness against Universal Text Perturbations on Large Language Models**

cs.CL

Accepted by ACL Findings 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.01873v2) [paper-pdf](http://arxiv.org/pdf/2406.01873v2)

**Authors**: Qian Lou, Xin Liang, Jiaqi Xue, Yancheng Zhang, Rui Xie, Mengxin Zheng

**Abstract**: It is imperative to ensure the stability of every prediction made by a language model; that is, a language's prediction should remain consistent despite minor input variations, like word substitutions. In this paper, we investigate the problem of certifying a language model's robustness against Universal Text Perturbations (UTPs), which have been widely used in universal adversarial attacks and backdoor attacks. Existing certified robustness based on random smoothing has shown considerable promise in certifying the input-specific text perturbations (ISTPs), operating under the assumption that any random alteration of a sample's clean or adversarial words would negate the impact of sample-wise perturbations. However, with UTPs, masking only the adversarial words can eliminate the attack. A naive method is to simply increase the masking ratio and the likelihood of masking attack tokens, but it leads to a significant reduction in both certified accuracy and the certified radius due to input corruption by extensive masking. To solve this challenge, we introduce a novel approach, the superior prompt search method, designed to identify a superior prompt that maintains higher certified accuracy under extensive masking. Additionally, we theoretically motivate why ensembles are a particularly suitable choice as base prompts for random smoothing. The method is denoted by superior prompt ensembling technique. We also empirically confirm this technique, obtaining state-of-the-art results in multiple settings. These methodologies, for the first time, enable high certified accuracy against both UTPs and ISTPs. The source code of CR-UTP is available at \url {https://github.com/UCFML-Research/CR-UTP}.



## **18. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

cs.LG

ICML 2024 Oral

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.12336v2) [paper-pdf](http://arxiv.org/pdf/2402.12336v2)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many large vision-language models (LVLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (LVLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of LVLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the down-stream LVLMs is required. The code and robust models are available at https://github.com/chs20/RobustVLM



## **19. Can Implicit Bias Imply Adversarial Robustness?**

cs.LG

icml 2024 camera-ready

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.15942v2) [paper-pdf](http://arxiv.org/pdf/2405.15942v2)

**Authors**: Hancheng Min, René Vidal

**Abstract**: The implicit bias of gradient-based training algorithms has been considered mostly beneficial as it leads to trained networks that often generalize well. However, Frei et al. (2023) show that such implicit bias can harm adversarial robustness. Specifically, they show that if the data consists of clusters with small inter-cluster correlation, a shallow (two-layer) ReLU network trained by gradient flow generalizes well, but it is not robust to adversarial attacks of small radius. Moreover, this phenomenon occurs despite the existence of a much more robust classifier that can be explicitly constructed from a shallow network. In this paper, we extend recent analyses of neuron alignment to show that a shallow network with a polynomial ReLU activation (pReLU) trained by gradient flow not only generalizes well but is also robust to adversarial attacks. Our results highlight the importance of the interplay between data structure and architecture design in the implicit bias and robustness of trained networks.



## **20. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03230v1) [paper-pdf](http://arxiv.org/pdf/2406.03230v1)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply an established methodology for analyzing distinctive activation patterns in the residual streams for a novel result of attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **21. Graph Neural Network Explanations are Fragile**

cs.CR

17 pages, 64 figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03193v1) [paper-pdf](http://arxiv.org/pdf/2406.03193v1)

**Authors**: Jiate Li, Meng Pang, Yun Dong, Jinyuan Jia, Binghui Wang

**Abstract**: Explainable Graph Neural Network (GNN) has emerged recently to foster the trust of using GNNs. Existing GNN explainers are developed from various perspectives to enhance the explanation performance. We take the first step to study GNN explainers under adversarial attack--We found that an adversary slightly perturbing graph structure can ensure GNN model makes correct predictions, but the GNN explainer yields a drastically different explanation on the perturbed graph. Specifically, we first formulate the attack problem under a practical threat model (i.e., the adversary has limited knowledge about the GNN explainer and a restricted perturbation budget). We then design two methods (i.e., one is loss-based and the other is deduction-based) to realize the attack. We evaluate our attacks on various GNN explainers and the results show these explainers are fragile.



## **22. Reconstructing training data from document understanding models**

cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03182v1) [paper-pdf](http://arxiv.org/pdf/2406.03182v1)

**Authors**: Jérémie Dentan, Arnaud Paran, Aymen Shabou

**Abstract**: Document understanding models are increasingly employed by companies to supplant humans in processing sensitive documents, such as invoices, tax notices, or even ID cards. However, the robustness of such models to privacy attacks remains vastly unexplored. This paper presents CDMI, the first reconstruction attack designed to extract sensitive fields from the training data of these models. We attack LayoutLM and BROS architectures, demonstrating that an adversary can perfectly reconstruct up to 4.1% of the fields of the documents used for fine-tuning, including some names, dates, and invoice amounts up to six-digit numbers. When our reconstruction attack is combined with a membership inference attack, our attack accuracy escalates to 22.5%. In addition, we introduce two new end-to-end metrics and evaluate our approach under various conditions: unimodal or bimodal data, LayoutLM or BROS backbones, four fine-tuning tasks, and two public datasets (FUNSD and SROIE). We also investigate the interplay between overfitting, predictive performance, and susceptibility to our attack. We conclude with a discussion on possible defenses against our attack and potential future research directions to construct robust document understanding models.



## **23. ZeroPur: Succinct Training-Free Adversarial Purification**

cs.CV

16 pages, 5 figures, under review

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03143v1) [paper-pdf](http://arxiv.org/pdf/2406.03143v1)

**Authors**: Xiuli Bi, Zonglin Yang, Bo Liu, Xiaodong Cun, Chi-Man Pun, Pietro Lio, Bin Xiao

**Abstract**: Adversarial purification is a kind of defense technique that can defend various unseen adversarial attacks without modifying the victim classifier. Existing methods often depend on external generative models or cooperation between auxiliary functions and victim classifiers. However, retraining generative models, auxiliary functions, or victim classifiers relies on the domain of the fine-tuned dataset and is computation-consuming. In this work, we suppose that adversarial images are outliers of the natural image manifold and the purification process can be considered as returning them to this manifold. Following this assumption, we present a simple adversarial purification method without further training to purify adversarial images, called ZeroPur. ZeroPur contains two steps: given an adversarial example, Guided Shift obtains the shifted embedding of the adversarial example by the guidance of its blurred counterparts; after that, Adaptive Projection constructs a directional vector by this shifted embedding to provide momentum, projecting adversarial images onto the manifold adaptively. ZeroPur is independent of external models and requires no retraining of victim classifiers or auxiliary functions, relying solely on victim classifiers themselves to achieve purification. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) using various classifier architectures (ResNet, WideResNet) demonstrate that our method achieves state-of-the-art robust performance. The code will be publicly available.



## **24. VQUNet: Vector Quantization U-Net for Defending Adversarial Atacks by Regularizing Unwanted Noise**

cs.CV

8 pages, 6 figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03117v1) [paper-pdf](http://arxiv.org/pdf/2406.03117v1)

**Authors**: Zhixun He, Mukesh Singhal

**Abstract**: Deep Neural Networks (DNN) have become a promising paradigm when developing Artificial Intelligence (AI) and Machine Learning (ML) applications. However, DNN applications are vulnerable to fake data that are crafted with adversarial attack algorithms. Under adversarial attacks, the prediction accuracy of DNN applications suffers, making them unreliable. In order to defend against adversarial attacks, we introduce a novel noise-reduction procedure, Vector Quantization U-Net (VQUNet), to reduce adversarial noise and reconstruct data with high fidelity. VQUNet features a discrete latent representation learning through a multi-scale hierarchical structure for both noise reduction and data reconstruction. The empirical experiments show that the proposed VQUNet provides better robustness to the target DNN models, and it outperforms other state-of-the-art noise-reduction-based defense methods under various adversarial attacks for both Fashion-MNIST and CIFAR10 datasets. When there is no adversarial attack, the defense method has less than 1% accuracy degradation for both datasets.



## **25. Revisiting the Trade-off between Accuracy and Robustness via Weight Distribution of Filters**

cs.CV

Accepted by TPAMI2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2306.03430v4) [paper-pdf](http://arxiv.org/pdf/2306.03430v4)

**Authors**: Xingxing Wei, Shiji Zhao, Bo li

**Abstract**: Adversarial attacks have been proven to be potential threats to Deep Neural Networks (DNNs), and many methods are proposed to defend against adversarial attacks. However, while enhancing the robustness, the clean accuracy will decline to a certain extent, implying a trade-off existed between the accuracy and robustness. In this paper, to meet the trade-off problem, we theoretically explore the underlying reason for the difference of the filters' weight distribution between standard-trained and robust-trained models and then argue that this is an intrinsic property for static neural networks, thus they are difficult to fundamentally improve the accuracy and adversarial robustness at the same time. Based on this analysis, we propose a sample-wise dynamic network architecture named Adversarial Weight-Varied Network (AW-Net), which focuses on dealing with clean and adversarial examples with a "divide and rule" weight strategy. The AW-Net adaptively adjusts the network's weights based on regulation signals generated by an adversarial router, which is directly influenced by the input sample. Benefiting from the dynamic network architecture, clean and adversarial examples can be processed with different network weights, which provides the potential to enhance both accuracy and adversarial robustness. A series of experiments demonstrate that our AW-Net is architecture-friendly to handle both clean and adversarial examples and can achieve better trade-off performance than state-of-the-art robust models.



## **26. Enhancing the Resilience of Graph Neural Networks to Topological Perturbations in Sparse Graphs**

cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03097v1) [paper-pdf](http://arxiv.org/pdf/2406.03097v1)

**Authors**: Shuqi He, Jun Zhuang, Ding Wang, Luyao Peng, Jun Song

**Abstract**: Graph neural networks (GNNs) have been extensively employed in node classification. Nevertheless, recent studies indicate that GNNs are vulnerable to topological perturbations, such as adversarial attacks and edge disruptions. Considerable efforts have been devoted to mitigating these challenges. For example, pioneering Bayesian methodologies, including GraphSS and LlnDT, incorporate Bayesian label transitions and topology-based label sampling to strengthen the robustness of GNNs. However, GraphSS is hindered by slow convergence, while LlnDT faces challenges in sparse graphs. To overcome these limitations, we propose a novel label inference framework, TraTopo, which combines topology-driven label propagation, Bayesian label transitions, and link analysis via random walks. TraTopo significantly surpasses its predecessors on sparse graphs by utilizing random walk sampling, specifically targeting isolated nodes for link prediction, thus enhancing its effectiveness in topological sampling contexts. Additionally, TraTopo employs a shortest-path strategy to refine link prediction, thereby reducing predictive overhead and improving label inference accuracy. Empirical evaluations highlight TraTopo's superiority in node classification, significantly exceeding contemporary GCN models in accuracy.



## **27. On the Duality Between Sharpness-Aware Minimization and Adversarial Training**

cs.LG

ICML 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.15152v2) [paper-pdf](http://arxiv.org/pdf/2402.15152v2)

**Authors**: Yihao Zhang, Hangzhou He, Jingyu Zhu, Huanran Chen, Yifei Wang, Zeming Wei

**Abstract**: Adversarial Training (AT), which adversarially perturb the input samples during training, has been acknowledged as one of the most effective defenses against adversarial attacks, yet suffers from inevitably decreased clean accuracy. Instead of perturbing the samples, Sharpness-Aware Minimization (SAM) perturbs the model weights during training to find a more flat loss landscape and improve generalization. However, as SAM is designed for better clean accuracy, its effectiveness in enhancing adversarial robustness remains unexplored. In this work, considering the duality between SAM and AT, we investigate the adversarial robustness derived from SAM. Intriguingly, we find that using SAM alone can improve adversarial robustness. To understand this unexpected property of SAM, we first provide empirical and theoretical insights into how SAM can implicitly learn more robust features, and conduct comprehensive experiments to show that SAM can improve adversarial robustness notably without sacrificing any clean accuracy, shedding light on the potential of SAM to be a substitute for AT when accuracy comes at a higher priority. Code is available at https://github.com/weizeming/SAM_AT.



## **28. Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections**

cs.LG

21 pages

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03052v1) [paper-pdf](http://arxiv.org/pdf/2406.03052v1)

**Authors**: Zihan Luo, Hong Huang, Yongkang Zhou, Jiping Zhang, Nuo Chen

**Abstract**: Despite the remarkable capabilities demonstrated by Graph Neural Networks (GNNs) in graph-related tasks, recent research has revealed the fairness vulnerabilities in GNNs when facing malicious adversarial attacks. However, all existing fairness attacks require manipulating the connectivity between existing nodes, which may be prohibited in reality. To this end, we introduce a Node Injection-based Fairness Attack (NIFA), exploring the vulnerabilities of GNN fairness in such a more realistic setting. In detail, NIFA first designs two insightful principles for node injection operations, namely the uncertainty-maximization principle and homophily-increase principle, and then optimizes injected nodes' feature matrix to further ensure the effectiveness of fairness attacks. Comprehensive experiments on three real-world datasets consistently demonstrate that NIFA can significantly undermine the fairness of mainstream GNNs, even including fairness-aware GNNs, by injecting merely 1% of nodes. We sincerely hope that our work can stimulate increasing attention from researchers on the vulnerability of GNN fairness, and encourage the development of corresponding defense mechanisms.



## **29. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.14478v2) [paper-pdf](http://arxiv.org/pdf/2405.14478v2)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Ivan Tesfai Ogbu, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyse samples; and (iii) analysing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we propose SLIFER, a novel Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples resistance to analysis, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER leveraging content-injections attacks, and we show that, counter-intuitively, attacks are blocked more by YARA rules than dynamic analysis due to byte artifacts created while optimizing the adversarial strategy.



## **30. DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross Domain**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2309.14585

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03017v1) [paper-pdf](http://arxiv.org/pdf/2406.03017v1)

**Authors**: Jun Liu, Jiantao Zhou, Jiandian Zeng, Jinyu Tian

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (ASR) and good generalizability. We design a novel attack method based on a \textit{Hierarchical} \textbf{Di}sentangled \textbf{F}eature space and \textit{cross domain}, called \textbf{DifAttack++}, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack++ firstly disentangles an image's latent feature into an \textit{adversarial feature} (AF) and a \textit{visual feature} (VF) via an autoencoder equipped with our specially designed \textbf{H}ierarchical \textbf{D}ecouple-\textbf{F}usion (HDF) module, where the AF dominates the adversarial capability of an image, while the VF largely determines its visual appearance. We train such autoencoders for the clean and adversarial image domains respectively, meanwhile realizing feature disentanglement, by using pairs of clean images and their Adversarial Examples (AEs) generated from available surrogate models via white-box attack methods. Eventually, in the black-box attack stage, DifAttack++ iteratively optimizes the AF according to the query feedback from the victim model until a successful AE is generated, while keeping the VF unaltered. Extensive experimental results demonstrate that our method achieves superior ASR and query efficiency than SOTA methods, meanwhile exhibiting much better visual quality of AEs. The code is available at https://github.com/csjunjun/DifAttack.git.



## **31. ACE: A Model Poisoning Attack on Contribution Evaluation Methods in Federated Learning**

cs.CR

To appear in the 33rd USENIX Security Symposium, 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.20975v2) [paper-pdf](http://arxiv.org/pdf/2405.20975v2)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bo Li, Radha Poovendran

**Abstract**: In Federated Learning (FL), a set of clients collaboratively train a machine learning model (called global model) without sharing their local training data. The local training data of clients is typically non-i.i.d. and heterogeneous, resulting in varying contributions from individual clients to the final performance of the global model. In response, many contribution evaluation methods were proposed, where the server could evaluate the contribution made by each client and incentivize the high-contributing clients to sustain their long-term participation in FL. Existing studies mainly focus on developing new metrics or algorithms to better measure the contribution of each client. However, the security of contribution evaluation methods of FL operating in adversarial environments is largely unexplored. In this paper, we propose the first model poisoning attack on contribution evaluation methods in FL, termed ACE. Specifically, we show that any malicious client utilizing ACE could manipulate the parameters of its local model such that it is evaluated to have a high contribution by the server, even when its local training data is indeed of low quality. We perform both theoretical analysis and empirical evaluations of ACE. Theoretically, we show our design of ACE can effectively boost the malicious client's perceived contribution when the server employs the widely-used cosine distance metric to measure contribution. Empirically, our results show ACE effectively and efficiently deceive five state-of-the-art contribution evaluation methods. In addition, ACE preserves the accuracy of the final global models on testing inputs. We also explore six countermeasures to defend ACE. Our results show they are inadequate to thwart ACE, highlighting the urgent need for new defenses to safeguard the contribution evaluation methods in FL.



## **32. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

cs.RO

9 pages (7 content, 1 reference, 1 appendix). 6 figures, submitted to  the IEEE Robotics and Automation Letters (RA-L)

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2403.05666v2) [paper-pdf](http://arxiv.org/pdf/2403.05666v2)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method to assess the resilience of the Iterative Closest Point (ICP) algorithm via deep-learning-based attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms prior to deployments is of utmost importance. The ICP algorithm has become the standard for lidar-based localization. However, the pose estimate it produces can be greatly affected by corruption in the measurements. Corruption can arise from a variety of scenarios such as occlusions, adverse weather, or mechanical issues in the sensor. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP empirically, our method focuses on finding the maximum possible ICP pose error using perturbation-based adversarial attacks. The proposed attack induces significant pose errors on ICP and outperforms baselines more than 88% of the time across a wide range of scenarios. As an example application, we demonstrate that our attack can be used to identify areas on a map where ICP is particularly vulnerable to corruption in the measurements.



## **33. Nonlinear Transformations Against Unlearnable Datasets**

cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.02883v1) [paper-pdf](http://arxiv.org/pdf/2406.02883v1)

**Authors**: Thushari Hapuarachchi, Jing Lin, Kaiqi Xiong, Mohamed Rahouti, Gitte Ost

**Abstract**: Automated scraping stands out as a common method for collecting data in deep learning models without the authorization of data owners. Recent studies have begun to tackle the privacy concerns associated with this data collection method. Notable approaches include Deepconfuse, error-minimizing, error-maximizing (also known as adversarial poisoning), Neural Tangent Generalization Attack, synthetic, autoregressive, One-Pixel Shortcut, Self-Ensemble Protection, Entangled Features, Robust Error-Minimizing, Hypocritical, and TensorClog. The data generated by those approaches, called "unlearnable" examples, are prevented "learning" by deep learning models. In this research, we investigate and devise an effective nonlinear transformation framework and conduct extensive experiments to demonstrate that a deep neural network can effectively learn from the data/examples traditionally considered unlearnable produced by the above twelve approaches. The resulting approach improves the ability to break unlearnable data compared to the linear separable technique recently proposed by researchers. Specifically, our extensive experiments show that the improvement ranges from 0.34% to 249.59% for the unlearnable CIFAR10 datasets generated by those twelve data protection approaches, except for One-Pixel Shortcut. Moreover, the proposed framework achieves over 100% improvement of test accuracy for Autoregressive and REM approaches compared to the linear separable technique. Our findings suggest that these approaches are inadequate in preventing unauthorized uses of data in machine learning models. There is an urgent need to develop more robust protection mechanisms that effectively thwart an attacker from accessing data without proper authorization from the owners.



## **34. Improving the Adversarial Robustness for Speaker Verification by Self-Supervised Learning**

cs.SD

Accepted by TASLP

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2106.00273v4) [paper-pdf](http://arxiv.org/pdf/2106.00273v4)

**Authors**: Haibin Wu, Xu Li, Andy T. Liu, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstract**: Previous works have shown that automatic speaker verification (ASV) is seriously vulnerable to malicious spoofing attacks, such as replay, synthetic speech, and recently emerged adversarial attacks. Great efforts have been dedicated to defending ASV against replay and synthetic speech; however, only a few approaches have been explored to deal with adversarial attacks. All the existing approaches to tackle adversarial attacks for ASV require the knowledge for adversarial samples generation, but it is impractical for defenders to know the exact attack algorithms that are applied by the in-the-wild attackers. This work is among the first to perform adversarial defense for ASV without knowing the specific attack algorithms. Inspired by self-supervised learning models (SSLMs) that possess the merits of alleviating the superficial noise in the inputs and reconstructing clean samples from the interrupted ones, this work regards adversarial perturbations as one kind of noise and conducts adversarial defense for ASV by SSLMs. Specifically, we propose to perform adversarial defense from two perspectives: 1) adversarial perturbation purification and 2) adversarial perturbation detection. Experimental results show that our detection module effectively shields the ASV by detecting adversarial samples with an accuracy of around 80%. Moreover, since there is no common metric for evaluating the adversarial defense performance for ASV, this work also formalizes evaluation metrics for adversarial defense considering both purification and detection based approaches into account. We sincerely encourage future works to benchmark their approaches based on the proposed evaluation framework.



## **35. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

cs.LG

arXiv admin note: text overlap with arXiv:2112.08331 by other authors

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.12295v2) [paper-pdf](http://arxiv.org/pdf/2405.12295v2)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska, Tomasz Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which enable the processing of graph-structured data without relying on predefined graph structures, are gaining importance in an increasingly wide variety of applications. As these networks demonstrate proficiency across a range of tasks, they become lucrative targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. A large effort has been made to develop model-stealing attacks that focus on models trained with images and texts. However, little attention has been paid to GNNs trained on graph data. This paper introduces a novel method for unsupervised model-stealing attacks against inductive GNNs, based on graph contrasting learning and spectral graph augmentations to efficiently extract information from the target model. The proposed attack is thoroughly evaluated on six datasets. The results show that this approach demonstrates a higher level of efficiency compared to existing stealing attacks. More concretely, our attack outperforms the baseline on all benchmarks achieving higher fidelity and downstream accuracy of the stolen model while requiring fewer queries sent to the target model.



## **36. Auditing Privacy Mechanisms via Label Inference Attacks**

cs.LG

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02797v1) [paper-pdf](http://arxiv.org/pdf/2406.02797v1)

**Authors**: Róbert István Busa-Fekete, Travis Dick, Claudio Gentile, Andrés Muñoz Medina, Adam Smith, Marika Swanberg

**Abstract**: We propose reconstruction advantage measures to audit label privatization mechanisms. A reconstruction advantage measure quantifies the increase in an attacker's ability to infer the true label of an unlabeled example when provided with a private version of the labels in a dataset (e.g., aggregate of labels from different users or noisy labels output by randomized response), compared to an attacker that only observes the feature vectors, but may have prior knowledge of the correlation between features and labels. We consider two such auditing measures: one additive, and one multiplicative. These incorporate previous approaches taken in the literature on empirical auditing and differential privacy. The measures allow us to place a variety of proposed privatization schemes -- some differentially private, some not -- on the same footing. We analyze these measures theoretically under a distributional model which encapsulates reasonable adversarial settings. We also quantify their behavior empirically on real and simulated prediction tasks. Across a range of experimental settings, we find that differentially private schemes dominate or match the privacy-utility tradeoff of more heuristic approaches.



## **37. Proof-of-Learning with Incentive Security**

cs.CR

16 pages

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2404.09005v5) [paper-pdf](http://arxiv.org/pdf/2404.09005v5)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.



## **38. Tree Proof-of-Position Algorithms**

cs.DS

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.06761v2) [paper-pdf](http://arxiv.org/pdf/2405.06761v2)

**Authors**: Aida Manzano Kharman, Pietro Ferraro, Homayoun Hamedmoghadam, Robert Shorten

**Abstract**: We present a novel class of proof-of-position algorithms: Tree-Proof-of-Position (T-PoP). This algorithm is decentralised, collaborative and can be computed in a privacy preserving manner, such that agents do not need to reveal their position publicly. We make no assumptions of honest behaviour in the system, and consider varying ways in which agents may misbehave. Our algorithm is therefore resilient to highly adversarial scenarios. This makes it suitable for a wide class of applications, namely those in which trust in a centralised infrastructure may not be assumed, or high security risk scenarios. Our algorithm has a worst case quadratic runtime, making it suitable for hardware constrained IoT applications. We also provide a mathematical model that summarises T-PoP's performance for varying operating conditions. We then simulate T-PoP's behaviour with a large number of agent-based simulations, which are in complete agreement with our mathematical model, thus demonstrating its validity. T-PoP can achieve high levels of reliability and security by tuning its operating conditions, both in high and low density environments. Finally, we also present a mathematical model to probabilistically detect platooning attacks.



## **39. Rethinking the Vulnerabilities of Face Recognition Systems:From a Practical Perspective**

cs.CR

19 pages,version 2

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.12786v2) [paper-pdf](http://arxiv.org/pdf/2405.12786v2)

**Authors**: Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Changjiang Li, Ting Wang, Shouling Ji

**Abstract**: Face Recognition Systems (FRS) have increasingly integrated into critical applications, including surveillance and user authentication, highlighting their pivotal role in modern security systems. Recent studies have revealed vulnerabilities in FRS to adversarial (e.g., adversarial patch attacks) and backdoor attacks (e.g., training data poisoning), raising significant concerns about their reliability and trustworthiness. Previous studies primarily focus on traditional adversarial or backdoor attacks, overlooking the resource-intensive or privileged-manipulation nature of such threats, thus limiting their practical generalization, stealthiness, universality and robustness. Correspondingly, in this paper, we delve into the inherent vulnerabilities in FRS through user studies and preliminary explorations. By exploiting these vulnerabilities, we identify a novel attack, facial identity backdoor attack dubbed FIBA, which unveils a potentially more devastating threat against FRS:an enrollment-stage backdoor attack. FIBA circumvents the limitations of traditional attacks, enabling broad-scale disruption by allowing any attacker donning a specific trigger to bypass these systems. This implies that after a single, poisoned example is inserted into the database, the corresponding trigger becomes a universal key for any attackers to spoof the FRS. This strategy essentially challenges the conventional attacks by initiating at the enrollment stage, dramatically transforming the threat landscape by poisoning the feature database rather than the training data.



## **40. Advancing Generalized Transfer Attack with Initialization Derived Bilevel Optimization and Dynamic Sequence Truncation**

cs.LG

Accepted by IJCAI 2024. 10 pages

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02064v1) [paper-pdf](http://arxiv.org/pdf/2406.02064v1)

**Authors**: Yaohua Liu, Jiaxin Gao, Xuan Liu, Xianghao Jiao, Xin Fan, Risheng Liu

**Abstract**: Transfer attacks generate significant interest for real-world black-box applications by crafting transferable adversarial examples through surrogate models. Whereas, existing works essentially directly optimize the single-level objective w.r.t. the surrogate model, which always leads to poor interpretability of attack mechanism and limited generalization performance over unknown victim models. In this work, we propose the \textbf{B}il\textbf{E}vel \textbf{T}ransfer \textbf{A}ttac\textbf{K} (BETAK) framework by establishing an initialization derived bilevel optimization paradigm, which explicitly reformulates the nested constraint relationship between the Upper-Level (UL) pseudo-victim attacker and the Lower-Level (LL) surrogate attacker. Algorithmically, we introduce the Hyper Gradient Response (HGR) estimation as an effective feedback for the transferability over pseudo-victim attackers, and propose the Dynamic Sequence Truncation (DST) technique to dynamically adjust the back-propagation path for HGR and reduce computational overhead simultaneously. Meanwhile, we conduct detailed algorithmic analysis and provide convergence guarantee to support non-convexity of the LL surrogate attacker. Extensive evaluations demonstrate substantial improvement of BETAK (e.g., $\mathbf{53.41}$\% increase of attack success rates against IncRes-v$2_{ens}$) against different victims and defense methods in targeted and untargeted attack scenarios. The source code is available at https://github.com/callous-youth/BETAK.



## **41. Graph Adversarial Diffusion Convolution**

cs.LG

Accepted by ICML 2024

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02059v1) [paper-pdf](http://arxiv.org/pdf/2406.02059v1)

**Authors**: Songtao Liu, Jinghui Chen, Tianfan Fu, Lu Lin, Marinka Zitnik, Dinghao Wu

**Abstract**: This paper introduces a min-max optimization formulation for the Graph Signal Denoising (GSD) problem. In this formulation, we first maximize the second term of GSD by introducing perturbations to the graph structure based on Laplacian distance and then minimize the overall loss of the GSD. By solving the min-max optimization problem, we derive a new variant of the Graph Diffusion Convolution (GDC) architecture, called Graph Adversarial Diffusion Convolution (GADC). GADC differs from GDC by incorporating an additional term that enhances robustness against adversarial attacks on the graph structure and noise in node features. Moreover, GADC improves the performance of GDC on heterophilic graphs. Extensive experiments demonstrate the effectiveness of GADC across various datasets. Code is available at https://github.com/SongtaoLiu0823/GADC.



## **42. ASETF: A Novel Method for Jailbreak Attack on LLMs through Translate Suffix Embeddings**

cs.CL

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2402.16006v2) [paper-pdf](http://arxiv.org/pdf/2402.16006v2)

**Authors**: Hao Wang, Hao Li, Minlie Huang, Lei Sha

**Abstract**: The safety defense methods of Large language models(LLMs) stays limited because the dangerous prompts are manually curated to just few known attack types, which fails to keep pace with emerging varieties. Recent studies found that attaching suffixes to harmful instructions can hack the defense of LLMs and lead to dangerous outputs. However, similar to traditional text adversarial attacks, this approach, while effective, is limited by the challenge of the discrete tokens. This gradient based discrete optimization attack requires over 100,000 LLM calls, and due to the unreadable of adversarial suffixes, it can be relatively easily penetrated by common defense methods such as perplexity filters. To cope with this challenge, in this paper, we proposes an Adversarial Suffix Embedding Translation Framework (ASETF), aimed at transforming continuous adversarial suffix embeddings into coherent and understandable text. This method greatly reduces the computational overhead during the attack process and helps to automatically generate multiple adversarial samples, which can be used as data to strengthen LLMs security defense. Experimental evaluations were conducted on Llama2, Vicuna, and other prominent LLMs, employing harmful directives sourced from the Advbench dataset. The results indicate that our method significantly reduces the computation time of adversarial suffixes and achieves a much better attack success rate to existing techniques, while significantly enhancing the textual fluency of the prompts. In addition, our approach can be generalized into a broader method for generating transferable adversarial suffixes that can successfully attack multiple LLMs, even black-box LLMs, such as ChatGPT and Gemini.



## **43. SVASTIN: Sparse Video Adversarial Attack via Spatio-Temporal Invertible Neural Networks**

cs.CV

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.01894v1) [paper-pdf](http://arxiv.org/pdf/2406.01894v1)

**Authors**: Yi Pan, Jun-Jie Huang, Zihan Chen, Wentao Zhao, Ziyue Wang

**Abstract**: Robust and imperceptible adversarial video attack is challenging due to the spatial and temporal characteristics of videos. The existing video adversarial attack methods mainly take a gradient-based approach and generate adversarial videos with noticeable perturbations. In this paper, we propose a novel Sparse Adversarial Video Attack via Spatio-Temporal Invertible Neural Networks (SVASTIN) to generate adversarial videos through spatio-temporal feature space information exchanging. It consists of a Guided Target Video Learning (GTVL) module to balance the perturbation budget and optimization speed and a Spatio-Temporal Invertible Neural Network (STIN) module to perform spatio-temporal feature space information exchanging between a source video and the target feature tensor learned by GTVL module. Extensive experiments on UCF-101 and Kinetics-400 demonstrate that our proposed SVASTIN can generate adversarial examples with higher imperceptibility than the state-of-the-art methods with the higher fooling rate. Code is available at \href{https://github.com/Brittany-Chen/SVASTIN}{https://github.com/Brittany-Chen/SVASTIN}.



## **44. Adversarial Attacks on Combinatorial Multi-Armed Bandits**

cs.LG

28 pages, Accepted to ICML 2024

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2310.05308v2) [paper-pdf](http://arxiv.org/pdf/2310.05308v2)

**Authors**: Rishab Balasubramanian, Jiawei Li, Prasad Tadepalli, Huazheng Wang, Qingyun Wu, Haoyu Zhao

**Abstract**: We study reward poisoning attacks on Combinatorial Multi-armed Bandits (CMAB). We first provide a sufficient and necessary condition for the attackability of CMAB, a notion to capture the vulnerability and robustness of CMAB. The attackability condition depends on the intrinsic properties of the corresponding CMAB instance such as the reward distributions of super arms and outcome distributions of base arms. Additionally, we devise an attack algorithm for attackable CMAB instances. Contrary to prior understanding of multi-armed bandits, our work reveals a surprising fact that the attackability of a specific CMAB instance also depends on whether the bandit instance is known or unknown to the adversary. This finding indicates that adversarial attacks on CMAB are difficult in practice and a general attack strategy for any CMAB instance does not exist since the environment is mostly unknown to the adversary. We validate our theoretical findings via extensive experiments on real-world CMAB applications including probabilistic maximum covering problem, online minimum spanning tree, cascading bandits for online ranking, and online shortest path.



## **45. Reproducibility Study on Adversarial Attacks Against Robust Transformer Trackers**

cs.CV

Published in Transactions on Machine Learning Research (05/2024):  https://openreview.net/forum?id=FEEKR0Vl9s

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01765v1) [paper-pdf](http://arxiv.org/pdf/2406.01765v1)

**Authors**: Fatemeh Nourilenjan Nokabadi, Jean-François Lalonde, Christian Gagné

**Abstract**: New transformer networks have been integrated into object tracking pipelines and have demonstrated strong performance on the latest benchmarks. This paper focuses on understanding how transformer trackers behave under adversarial attacks and how different attacks perform on tracking datasets as their parameters change. We conducted a series of experiments to evaluate the effectiveness of existing adversarial attacks on object trackers with transformer and non-transformer backbones. We experimented on 7 different trackers, including 3 that are transformer-based, and 4 which leverage other architectures. These trackers are tested against 4 recent attack methods to assess their performance and robustness on VOT2022ST, UAV123 and GOT10k datasets. Our empirical study focuses on evaluating adversarial robustness of object trackers based on bounding box versus binary mask predictions, and attack methods at different levels of perturbations. Interestingly, our study found that altering the perturbation level may not significantly affect the overall object tracking results after the attack. Similarly, the sparsity and imperceptibility of the attack perturbations may remain stable against perturbation level shifts. By applying a specific attack on all transformer trackers, we show that new transformer trackers having a stronger cross-attention modeling achieve a greater adversarial robustness on tracking datasets, such as VOT2022ST and GOT10k. Our results also indicate the necessity for new attack methods to effectively tackle the latest types of transformer trackers. The codes necessary to reproduce this study are available at https://github.com/fatemehN/ReproducibilityStudy.



## **46. On the completeness of several fortification-interdiction games in the Polynomial Hierarchy**

cs.CC

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01756v1) [paper-pdf](http://arxiv.org/pdf/2406.01756v1)

**Authors**: Alberto Boggio Tomasaz, Margarida Carvalho, Roberto Cordone, Pierre Hosteins

**Abstract**: Fortification-interdiction games are tri-level adversarial games where two opponents act in succession to protect, disrupt and simply use an infrastructure for a specific purpose. Many such games have been formulated and tackled in the literature through specific algorithmic methods, however very few investigations exist on the completeness of such fortification problems in order to locate them rigorously in the polynomial hierarchy. We clarify the completeness status of several well-known fortification problems, such as the Tri-level Interdiction Knapsack Problem with unit fortification and attack weights, the Max-flow Interdiction Problem and Shortest Path Interdiction Problem with Fortification, the Multi-level Critical Node Problem with unit weights, as well as a well-studied electric grid defence planning problem. For all of these problems, we prove their completeness either for the $\Sigma^p_2$ or the $\Sigma^p_3$ class of the polynomial hierarchy. We also prove that the Multi-level Fortification-Interdiction Knapsack Problem with an arbitrary number of protection and interdiction rounds and unit fortification and attack weights is complete for any level of the polynomial hierarchy, therefore providing a useful basis for further attempts at proving the completeness of protection-interdiction games at any level of said hierarchy.



## **47. MAWSEO: Adversarial Wiki Search Poisoning for Illicit Online Promotion**

cs.CR

Accepted at the 45th IEEE Symposium on Security and Privacy (IEEE S&P  2024)

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2304.11300v3) [paper-pdf](http://arxiv.org/pdf/2304.11300v3)

**Authors**: Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang, Xiaozhong Liu

**Abstract**: As a prominent instance of vandalism edits, Wiki search poisoning for illicit promotion is a cybercrime in which the adversary aims at editing Wiki articles to promote illicit businesses through Wiki search results of relevant queries. In this paper, we report a study that, for the first time, shows that such stealthy blackhat SEO on Wiki can be automated. Our technique, called MAWSEO, employs adversarial revisions to achieve real-world cybercriminal objectives, including rank boosting, vandalism detection evasion, topic relevancy, semantic consistency, user awareness (but not alarming) of promotional content, etc. Our evaluation and user study demonstrate that MAWSEO is capable of effectively and efficiently generating adversarial vandalism edits, which can bypass state-of-the-art built-in Wiki vandalism detectors, and also get promotional content through to Wiki users without triggering their alarms. In addition, we investigated potential defense, including coherence based detection and adversarial training of vandalism detection, against our attack in the Wiki ecosystem.



## **48. Mixing Classifiers to Alleviate the Accuracy-Robustness Trade-Off**

cs.LG

arXiv admin note: text overlap with arXiv:2301.12554

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2311.15165v2) [paper-pdf](http://arxiv.org/pdf/2311.15165v2)

**Authors**: Yatong Bai, Brendon G. Anderson, Somayeh Sojoudi

**Abstract**: Deep neural classifiers have recently found tremendous success in data-driven control systems. However, existing models suffer from a trade-off between accuracy and adversarial robustness. This limitation must be overcome in the control of safety-critical systems that require both high performance and rigorous robustness guarantees. In this work, we develop classifiers that simultaneously inherit high robustness from robust models and high accuracy from standard models. Specifically, we propose a theoretically motivated formulation that mixes the output probabilities of a standard neural network and a robust neural network. Both base classifiers are pre-trained, and thus our method does not require additional training. Our numerical experiments verify that the mixed classifier noticeably improves the accuracy-robustness trade-off and identify the confidence property of the robust base classifier as the key leverage of this more benign trade-off. Our theoretical results prove that under mild assumptions, when the robustness of the robust base model is certifiable, no alteration or attack within a closed-form $\ell_p$ radius on an input can result in the misclassification of the mixed classifier.



## **49. Model for Peanuts: Hijacking ML Models without Training Access is Possible**

cs.CR

17 pages, 14 figures, 7 tables

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01708v1) [paper-pdf](http://arxiv.org/pdf/2406.01708v1)

**Authors**: Mahmoud Ghorbel, Halima Bouzidi, Ioan Marius Bilasco, Ihsen Alouani

**Abstract**: The massive deployment of Machine Learning (ML) models has been accompanied by the emergence of several attacks that threaten their trustworthiness and raise ethical and societal concerns such as invasion of privacy, discrimination risks, and lack of accountability. Model hijacking is one of these attacks, where the adversary aims to hijack a victim model to execute a different task than its original one. Model hijacking can cause accountability and security risks since a hijacked model owner can be framed for having their model offering illegal or unethical services. Prior state-of-the-art works consider model hijacking as a training time attack, whereby an adversary requires access to the ML model training to execute their attack. In this paper, we consider a stronger threat model where the attacker has no access to the training phase of the victim model. Our intuition is that ML models, typically over-parameterized, might (unintentionally) learn more than the intended task for they are trained. We propose a simple approach for model hijacking at inference time named SnatchML to classify unknown input samples using distance measures in the latent space of the victim model to previously known samples associated with the hijacking task classes. SnatchML empirically shows that benign pre-trained models can execute tasks that are semantically related to the initial task. Surprisingly, this can be true even for hijacking tasks unrelated to the original task. We also explore different methods to mitigate this risk. We first propose a novel approach we call meta-unlearning, designed to help the model unlearn a potentially malicious task while training on the original task dataset. We also provide insights on over-parameterization as one possible inherent factor that makes model hijacking easier, and we accordingly propose a compression-based countermeasure against this attack.



## **50. From Feature Visualization to Visual Circuits: Effect of Adversarial Model Manipulation**

cs.CV

Under review

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01365v1) [paper-pdf](http://arxiv.org/pdf/2406.01365v1)

**Authors**: Geraldin Nanfack, Michael Eickenberg, Eugene Belilovsky

**Abstract**: Understanding the inner working functionality of large-scale deep neural networks is challenging yet crucial in several high-stakes applications. Mechanistic inter- pretability is an emergent field that tackles this challenge, often by identifying human-understandable subgraphs in deep neural networks known as circuits. In vision-pretrained models, these subgraphs are usually interpreted by visualizing their node features through a popular technique called feature visualization. Recent works have analyzed the stability of different feature visualization types under the adversarial model manipulation framework. This paper starts by addressing limitations in existing works by proposing a novel attack called ProxPulse that simultaneously manipulates the two types of feature visualizations. Surprisingly, when analyzing these attacks under the umbrella of visual circuits, we find that visual circuits show some robustness to ProxPulse. We, therefore, introduce a new attack based on ProxPulse that unveils the manipulability of visual circuits, shedding light on their lack of robustness. The effectiveness of these attacks is validated using pre-trained AlexNet and ResNet-50 models on ImageNet.



