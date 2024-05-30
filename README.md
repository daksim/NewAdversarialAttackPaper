# Latest Adversarial Attack Papers
**update at 2024-05-30 19:10:48**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning**

cs.CV

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19237v1) [paper-pdf](http://arxiv.org/pdf/2405.19237v1)

**Authors**: Ruchika Chavhan, Da Li, Timothy Hospedales

**Abstract**: While large-scale text-to-image diffusion models have demonstrated impressive image-generation capabilities, there are significant concerns about their potential misuse for generating unsafe content, violating copyright, and perpetuating societal biases. Recently, the text-to-image generation community has begun addressing these concerns by editing or unlearning undesired concepts from pre-trained models. However, these methods often involve data-intensive and inefficient fine-tuning or utilize various forms of token remapping, rendering them susceptible to adversarial jailbreaks. In this paper, we present a simple and effective training-free approach, ConceptPrune, wherein we first identify critical regions within pre-trained models responsible for generating undesirable concepts, thereby facilitating straightforward concept unlearning via weight pruning. Experiments across a range of concepts including artistic styles, nudity, object erasure, and gender debiasing demonstrate that target concepts can be efficiently erased by pruning a tiny fraction, approximately 0.12% of total weights, enabling multi-concept erasure and robustness against various white-box and black-box adversarial attacks.



## **2. Gone but Not Forgotten: Improved Benchmarks for Machine Unlearning**

cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19211v1) [paper-pdf](http://arxiv.org/pdf/2405.19211v1)

**Authors**: Keltin Grimes, Collin Abidi, Cole Frank, Shannon Gallagher

**Abstract**: Machine learning models are vulnerable to adversarial attacks, including attacks that leak information about the model's training data. There has recently been an increase in interest about how to best address privacy concerns, especially in the presence of data-removal requests. Machine unlearning algorithms aim to efficiently update trained models to comply with data deletion requests while maintaining performance and without having to resort to retraining the model from scratch, a costly endeavor. Several algorithms in the machine unlearning literature demonstrate some level of privacy gains, but they are often evaluated only on rudimentary membership inference attacks, which do not represent realistic threats. In this paper we describe and propose alternative evaluation methods for three key shortcomings in the current evaluation of unlearning algorithms. We show the utility of our alternative evaluations via a series of experiments of state-of-the-art unlearning algorithms on different computer vision datasets, presenting a more detailed picture of the state of the field.



## **3. Model Agnostic Defense against Adversarial Patch Attacks on Object Detection in Unmanned Aerial Vehicles**

cs.CV

submitted to IROS 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19179v1) [paper-pdf](http://arxiv.org/pdf/2405.19179v1)

**Authors**: Saurabh Pathak, Samridha Shrestha, Abdelrahman AlMahmoud

**Abstract**: Object detection forms a key component in Unmanned Aerial Vehicles (UAVs) for completing high-level tasks that depend on the awareness of objects on the ground from an aerial perspective. In that scenario, adversarial patch attacks on an onboard object detector can severely impair the performance of upstream tasks. This paper proposes a novel model-agnostic defense mechanism against the threat of adversarial patch attacks in the context of UAV-based object detection. We formulate adversarial patch defense as an occlusion removal task. The proposed defense method can neutralize adversarial patches located on objects of interest, without exposure to adversarial patches during training. Our lightweight single-stage defense approach allows us to maintain a model-agnostic nature, that once deployed does not require to be updated in response to changes in the object detection pipeline. The evaluations in digital and physical domains show the feasibility of our method for deployment in UAV object detection pipelines, by significantly decreasing the Attack Success Ratio without incurring significant processing costs. As a result, the proposed defense solution can improve the reliability of object detection for UAVs.



## **4. Introducing Adaptive Continuous Adversarial Training (ACAT) to Enhance ML Robustness**

cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2403.10461v2) [paper-pdf](http://arxiv.org/pdf/2403.10461v2)

**Authors**: Mohamed elShehaby, Aditya Kotha, Ashraf Matrawy

**Abstract**: Adversarial training enhances the robustness of Machine Learning (ML) models against adversarial attacks. However, obtaining labeled training and adversarial training data in network/cybersecurity domains is challenging and costly. Therefore, this letter introduces Adaptive Continuous Adversarial Training (ACAT), a method that integrates adversarial training samples into the model during continuous learning sessions using real-world detected adversarial data. Experimental results with a SPAM detection dataset demonstrate that ACAT reduces the time required for adversarial sample detection compared to traditional processes. Moreover, the accuracy of the under-attack ML-based SPAM filter increased from 69% to over 88% after just three retraining sessions.



## **5. Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior**

cs.LG

ICML 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19098v1) [paper-pdf](http://arxiv.org/pdf/2405.19098v1)

**Authors**: Shuyu Cheng, Yibo Miao, Yinpeng Dong, Xiao Yang, Xiao-Shan Gao, Jun Zhu

**Abstract**: This paper studies the challenging black-box adversarial attack that aims to generate adversarial examples against a black-box model by only using output feedback of the model to input queries. Some previous methods improve the query efficiency by incorporating the gradient of a surrogate white-box model into query-based attacks due to the adversarial transferability. However, the localized gradient is not informative enough, making these methods still query-intensive. In this paper, we propose a Prior-guided Bayesian Optimization (P-BO) algorithm that leverages the surrogate model as a global function prior in black-box adversarial attacks. As the surrogate model contains rich prior information of the black-box one, P-BO models the attack objective with a Gaussian process whose mean function is initialized as the surrogate model's loss. Our theoretical analysis on the regret bound indicates that the performance of P-BO may be affected by a bad prior. Therefore, we further propose an adaptive integration strategy to automatically adjust a coefficient on the function prior by minimizing the regret bound. Extensive experiments on image classifiers and large vision-language models demonstrate the superiority of the proposed algorithm in reducing queries and improving attack success rates compared with the state-of-the-art black-box attacks. Code is available at https://github.com/yibo-miao/PBO-Attack.



## **6. New perspectives on the optimal placement of detectors for suicide bombers using metaheuristics**

cs.NE

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19060v1) [paper-pdf](http://arxiv.org/pdf/2405.19060v1)

**Authors**: Carlos Cotta, José E. Gallardo

**Abstract**: We consider an operational model of suicide bombing attacks -- an increasingly prevalent form of terrorism -- against specific targets, and the use of protective countermeasures based on the deployment of detectors over the area under threat. These detectors have to be carefully located in order to minimize the expected number of casualties or the economic damage suffered, resulting in a hard optimization problem for which different metaheuristics have been proposed. Rather than assuming random decisions by the attacker, the problem is approached by considering different models of the latter, whereby he takes informed decisions on which objective must be targeted and through which path it has to be reached based on knowledge on the importance or value of the objectives or on the defensive strategy of the defender (a scenario that can be regarded as an adversarial game). We consider four different algorithms, namely a greedy heuristic, a hill climber, tabu search and an evolutionary algorithm, and study their performance on a broad collection of problem instances trying to resemble different realistic settings such as a coastal area, a modern urban area, and the historic core of an old town. It is shown that the adversarial scenario is harder for all techniques, and that the evolutionary algorithm seems to adapt better to the complexity of the resulting search landscape.



## **7. Verifiably Robust Conformal Prediction**

cs.LO

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18942v1) [paper-pdf](http://arxiv.org/pdf/2405.18942v1)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces \emph{VRCP (Verifiably Robust Conformal Prediction)}, a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.



## **8. Proactive Load-Shaping Strategies with Privacy-Cost Trade-offs in Residential Households based on Deep Reinforcement Learning**

eess.SY

7 pages

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18888v1) [paper-pdf](http://arxiv.org/pdf/2405.18888v1)

**Authors**: Ruichang Zhang, Youcheng Sun, Mustafa A. Mustafa

**Abstract**: Smart meters play a crucial role in enhancing energy management and efficiency, but they raise significant privacy concerns by potentially revealing detailed user behaviors through energy consumption patterns. Recent scholarly efforts have focused on developing battery-aided load-shaping techniques to protect user privacy while balancing costs. This paper proposes a novel deep reinforcement learning-based load-shaping algorithm (PLS-DQN) designed to protect user privacy by proactively creating artificial load signatures that mislead potential attackers. We evaluate our proposed algorithm against a non-intrusive load monitoring (NILM) adversary. The results demonstrate that our approach not only effectively conceals real energy usage patterns but also outperforms state-of-the-art methods in enhancing user privacy while maintaining cost efficiency.



## **9. Enhancing Security and Privacy in Federated Learning using Update Digests and Voting-Based Defense**

cs.CR

14 pages

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18802v1) [paper-pdf](http://arxiv.org/pdf/2405.18802v1)

**Authors**: Wenjie Li, Kai Fan, Jingyuan Zhang, Hui Li, Wei Yang Bryan Lim, Qiang Yang

**Abstract**: Federated Learning (FL) is a promising privacy-preserving machine learning paradigm that allows data owners to collaboratively train models while keeping their data localized. Despite its potential, FL faces challenges related to the trustworthiness of both clients and servers, especially in the presence of curious or malicious adversaries. In this paper, we introduce a novel framework named \underline{\textbf{F}}ederated \underline{\textbf{L}}earning with \underline{\textbf{U}}pdate \underline{\textbf{D}}igest (FLUD), which addresses the critical issues of privacy preservation and resistance to Byzantine attacks within distributed learning environments. FLUD utilizes an innovative approach, the $\mathsf{LinfSample}$ method, allowing clients to compute the $l_{\infty}$ norm across sliding windows of updates as an update digest. This digest enables the server to calculate a shared distance matrix, significantly reducing the overhead associated with Secure Multi-Party Computation (SMPC) by three orders of magnitude while effectively distinguishing between benign and malicious updates. Additionally, FLUD integrates a privacy-preserving, voting-based defense mechanism that employs optimized SMPC protocols to minimize communication rounds. Our comprehensive experiments demonstrate FLUD's effectiveness in countering Byzantine adversaries while incurring low communication and runtime overhead. FLUD offers a scalable framework for secure and reliable FL in distributed environments, facilitating its application in scenarios requiring robust data management and security.



## **10. MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training**

cs.CR

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2311.00919v2) [paper-pdf](http://arxiv.org/pdf/2311.00919v2)

**Authors**: Jiacheng Li, Ninghui Li, Bruno Ribeiro

**Abstract**: In Member Inference (MI) attacks, the adversary try to determine whether an instance is used to train a machine learning (ML) model. MI attacks are a major privacy concern when using private data to train ML models. Most MI attacks in the literature take advantage of the fact that ML models are trained to fit the training data well, and thus have very low loss on training instances. Most defenses against MI attacks therefore try to make the model fit the training data less well. Doing so, however, generally results in lower accuracy. We observe that training instances have different degrees of vulnerability to MI attacks. Most instances will have low loss even when not included in training. For these instances, the model can fit them well without concerns of MI attacks. An effective defense only needs to (possibly implicitly) identify instances that are vulnerable to MI attacks and avoids overfitting them. A major challenge is how to achieve such an effect in an efficient training process. Leveraging two distinct recent advancements in representation learning: counterfactually-invariant representations and subspace learning methods, we introduce a novel Membership-Invariant Subspace Training (MIST) method to defend against MI attacks. MIST avoids overfitting the vulnerable instances without significant impact on other instances. We have conducted extensive experimental studies, comparing MIST with various other state-of-the-art (SOTA) MI defenses against several SOTA MI attacks. We find that MIST outperforms other defenses while resulting in minimal reduction in testing accuracy.



## **11. Leveraging Many-To-Many Relationships for Defending Against Visual-Language Adversarial Attacks**

cs.CV

Under review

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18770v1) [paper-pdf](http://arxiv.org/pdf/2405.18770v1)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos

**Abstract**: Recent studies have revealed that vision-language (VL) models are vulnerable to adversarial attacks for image-text retrieval (ITR). However, existing defense strategies for VL models primarily focus on zero-shot image classification, which do not consider the simultaneous manipulation of image and text, as well as the inherent many-to-many (N:N) nature of ITR, where a single image can be described in numerous ways, and vice versa. To this end, this paper studies defense strategies against adversarial attacks on VL models for ITR for the first time. Particularly, we focus on how to leverage the N:N relationship in ITR to enhance adversarial robustness. We found that, although adversarial training easily overfits to specific one-to-one (1:1) image-text pairs in the train data, diverse augmentation techniques to create one-to-many (1:N) / many-to-one (N:1) image-text pairs can significantly improve adversarial robustness in VL models. Additionally, we show that the alignment of the augmented image-text pairs is crucial for the effectiveness of the defense strategy, and that inappropriate augmentations can even degrade the model's performance. Based on these findings, we propose a novel defense strategy that leverages the N:N relationship in ITR, which effectively generates diverse yet highly-aligned N:N pairs using basic augmentations and generative model-based augmentations. This work provides a novel perspective on defending against adversarial attacks in VL tasks and opens up new research directions for future work.



## **12. Genshin: General Shield for Natural Language Processing with Large Language Models**

cs.CL

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18741v1) [paper-pdf](http://arxiv.org/pdf/2405.18741v1)

**Authors**: Xiao Peng, Tao Liu, Ying Wang

**Abstract**: Large language models (LLMs) like ChatGPT, Gemini, or LLaMA have been trending recently, demonstrating considerable advancement and generalizability power in countless domains. However, LLMs create an even bigger black box exacerbating opacity, with interpretability limited to few approaches. The uncertainty and opacity embedded in LLMs' nature restrict their application in high-stakes domains like financial fraud, phishing, etc. Current approaches mainly rely on traditional textual classification with posterior interpretable algorithms, suffering from attackers who may create versatile adversarial samples to break the system's defense, forcing users to make trade-offs between efficiency and robustness. To address this issue, we propose a novel cascading framework called Genshin (General Shield for Natural Language Processing with Large Language Models), utilizing LLMs as defensive one-time plug-ins. Unlike most applications of LLMs that try to transform text into something new or structural, Genshin uses LLMs to recover text to its original state. Genshin aims to combine the generalizability of the LLM, the discrimination of the median model, and the interpretability of the simple model. Our experiments on the task of sentimental analysis and spam detection have shown fatal flaws of the current median models and exhilarating results on LLMs' recovery ability, demonstrating that Genshin is both effective and efficient. In our ablation study, we unearth several intriguing observations. Utilizing the LLM defender, a tool derived from the 4th paradigm, we have reproduced BERT's 15% optimal mask rate results in the 3rd paradigm of NLP. Additionally, when employing the LLM as a potential adversarial tool, attackers are capable of executing effective attacks that are nearly semantically lossless.



## **13. Security--Throughput Tradeoff of Nakamoto Consensus under Bandwidth Constraints**

cs.CR

ACM Conference on Computer and Communications Security 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2303.09113v3) [paper-pdf](http://arxiv.org/pdf/2303.09113v3)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: For Nakamoto's longest-chain consensus protocol, whose proof-of-work (PoW) and proof-of-stake (PoS) variants power major blockchains such as Bitcoin and Cardano, we revisit the classic problem of the security-performance tradeoff: Given a network of nodes with limited capacities, against what fraction of adversary power is Nakamoto consensus (NC) secure for a given block production rate? State-of-the-art analyses of Nakamoto's protocol fail to answer this question because their bounded-delay model does not capture realistic constraints such as limited communication- and computation-resources. We develop a new analysis technique to prove a refined security-performance tradeoff for PoW Nakamoto consensus in a bounded-bandwidth model. In this model, we show that, in contrast to the classic bounded-delay model, Nakamoto's private attack is no longer the worst attack, and a new attack strategy we call the teasing strategy, that exploits the network congestion caused by limited bandwidth, is strictly worse. In PoS, equivocating blocks can exacerbate congestion, making the traditional PoS Nakamoto consensus protocol insecure except at very low block production rates. To counter such equivocation spamming, we present a variant of the PoS NC protocol we call Blanking NC (BlaNC), which achieves the same resilience as PoW NC.



## **14. Watermarking Counterfactual Explanations**

cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18671v1) [paper-pdf](http://arxiv.org/pdf/2405.18671v1)

**Authors**: Hangzhi Guo, Amulya Yadav

**Abstract**: The field of Explainable Artificial Intelligence (XAI) focuses on techniques for providing explanations to end-users about the decision-making processes that underlie modern-day machine learning (ML) models. Within the vast universe of XAI techniques, counterfactual (CF) explanations are often preferred by end-users as they help explain the predictions of ML models by providing an easy-to-understand & actionable recourse (or contrastive) case to individual end-users who are adversely impacted by predicted outcomes. However, recent studies have shown significant security concerns with using CF explanations in real-world applications; in particular, malicious adversaries can exploit CF explanations to perform query-efficient model extraction attacks on proprietary ML models. In this paper, we propose a model-agnostic watermarking framework (for adding watermarks to CF explanations) that can be leveraged to detect unauthorized model extraction attacks (which rely on the watermarked CF explanations). Our novel framework solves a bi-level optimization problem to embed an indistinguishable watermark into the generated CF explanation such that any future model extraction attacks that rely on these watermarked CF explanations can be detected using a null hypothesis significance testing (NHST) scheme, while ensuring that these embedded watermarks do not compromise the quality of the generated CF explanations. We evaluate this framework's performance across a diverse set of real-world datasets, CF explanation methods, and model extraction techniques, and show that our watermarking detection system can be used to accurately identify extracted ML models that are trained using the watermarked CF explanations. Our work paves the way for the secure adoption of CF explanations in real-world applications.



## **15. PureGen: Universal Data Purification for Train-Time Poison Defense via Generative Model Dynamics**

cs.LG

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18627v1) [paper-pdf](http://arxiv.org/pdf/2405.18627v1)

**Authors**: Sunay Bhat, Jeffrey Jiang, Omead Pooladzandi, Alexander Branch, Gregory Pottie

**Abstract**: Train-time data poisoning attacks threaten machine learning models by introducing adversarial examples during training, leading to misclassification. Current defense methods often reduce generalization performance, are attack-specific, and impose significant training overhead. To address this, we introduce a set of universal data purification methods using a stochastic transform, $\Psi(x)$, realized via iterative Langevin dynamics of Energy-Based Models (EBMs), Denoising Diffusion Probabilistic Models (DDPMs), or both. These approaches purify poisoned data with minimal impact on classifier generalization. Our specially trained EBMs and DDPMs provide state-of-the-art defense against various attacks (including Narcissus, Bullseye Polytope, Gradient Matching) on CIFAR-10, Tiny-ImageNet, and CINIC-10, without needing attack or classifier-specific information. We discuss performance trade-offs and show that our methods remain highly effective even with poisoned or distributionally shifted generative model training data.



## **16. Wavelet-Based Image Tokenizer for Vision Transformers**

cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18616v1) [paper-pdf](http://arxiv.org/pdf/2405.18616v1)

**Authors**: Zhenhai Zhu, Radu Soricut

**Abstract**: Non-overlapping patch-wise convolution is the default image tokenizer for all state-of-the-art vision Transformer (ViT) models. Even though many ViT variants have been proposed to improve its efficiency and accuracy, little research on improving the image tokenizer itself has been reported in the literature. In this paper, we propose a new image tokenizer based on wavelet transformation. We show that ViT models with the new tokenizer achieve both higher training throughput and better top-1 precision for the ImageNet validation set. We present a theoretical analysis on why the proposed tokenizer improves the training throughput without any change to ViT model architecture. Our analysis suggests that the new tokenizer can effectively handle high-resolution images and is naturally resistant to adversarial attack. Furthermore, the proposed image tokenizer offers a fresh perspective on important new research directions for ViT-based model design, such as image tokens on a non-uniform grid for image understanding.



## **17. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2402.17012v2) [paper-pdf](http://arxiv.org/pdf/2402.17012v2)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model which leverages recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Taken together, these results represent the strongest existing privacy attacks against both pretrained and fine-tuned LLMs for MIAs and training data extraction, which are of independent scientific interest and have important practical implications for LLM security, privacy, and copyright issues.



## **18. Unleashing the potential of prompt engineering: a comprehensive review**

cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2310.14735v3) [paper-pdf](http://arxiv.org/pdf/2310.14735v3)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review explores the transformative potential of prompt engineering within the realm of large language models (LLMs) and multimodal language models (MMLMs). The development of AI, from its inception in the 1950s to the emergence of neural networks and deep learning architectures, has culminated in sophisticated LLMs like GPT-4 and BERT, as well as MMLMs like DALL-E and CLIP. These models have revolutionized tasks in diverse fields such as workplace automation, healthcare, and education. Prompt engineering emerges as a crucial technique to maximize the utility and accuracy of these models. This paper delves into both foundational and advanced methodologies of prompt engineering, including techniques like Chain of Thought, Self-consistency, and Generated Knowledge, which significantly enhance model performance. Additionally, it examines the integration of multimodal data through innovative approaches such as Multi-modal Prompt Learning (MaPLe), Conditional Prompt Learning, and Context Optimization. Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review underscores the pivotal role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.



## **19. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

cs.AI

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18166v1) [paper-pdf](http://arxiv.org/pdf/2405.18166v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.



## **20. Exploiting LLM Quantization**

cs.LG

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18137v1) [paper-pdf](http://arxiv.org/pdf/2405.18137v1)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.



## **21. S-Eval: Automatic and Adaptive Test Generation for Benchmarking Safety Evaluation of Large Language Models**

cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.14191v3) [paper-pdf](http://arxiv.org/pdf/2405.14191v3)

**Authors**: Xiaohan Yuan, Jinfeng Li, Dongxia Wang, Yuefeng Chen, Xiaofeng Mao, Longtao Huang, Hui Xue, Wenhai Wang, Kui Ren, Jingyi Wang

**Abstract**: Large Language Models have gained considerable attention for their revolutionary capabilities. However, there is also growing concern on their safety implications, making a comprehensive safety evaluation for LLMs urgently needed before model deployment. In this work, we propose S-Eval, a new comprehensive, multi-dimensional and open-ended safety evaluation benchmark. At the core of S-Eval is a novel LLM-based automatic test prompt generation and selection framework, which trains an expert testing LLM Mt combined with a range of test selection strategies to automatically construct a high-quality test suite for the safety evaluation. The key to the automation of this process is a novel expert safety-critique LLM Mc able to quantify the riskiness score of an LLM's response, and additionally produce risk tags and explanations. Besides, the generation process is also guided by a carefully designed risk taxonomy with four different levels, covering comprehensive and multi-dimensional safety risks of concern. Based on these, we systematically construct a new and large-scale safety evaluation benchmark for LLMs consisting of 220,000 evaluation prompts, including 20,000 base risk prompts (10,000 in Chinese and 10,000 in English) and 200,000 corresponding attack prompts derived from 10 popular adversarial instruction attacks against LLMs. Moreover, considering the rapid evolution of LLMs and accompanied safety threats, S-Eval can be flexibly configured and adapted to include new risks, attacks and models. S-Eval is extensively evaluated on 20 popular and representative LLMs. The results confirm that S-Eval can better reflect and inform the safety risks of LLMs compared to existing benchmarks. We also explore the impacts of parameter scales, language environments, and decoding parameters on the evaluation, providing a systematic methodology for evaluating the safety of LLMs.



## **22. Towards Unified Robustness Against Both Backdoor and Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.17929v1) [paper-pdf](http://arxiv.org/pdf/2405.17929v1)

**Authors**: Zhenxing Niu, Yuyao Sun, Qiguang Miao, Rong Jin, Gang Hua

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to both backdoor and adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct robustness problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, this paper revealed that there is an intriguing connection between them: (1) planting a backdoor into a model will significantly affect the model's adversarial examples; (2) for an infected model, its adversarial examples have similar features as the triggered images. Based on these observations, a novel Progressive Unified Defense (PUD) algorithm is proposed to defend against backdoor and adversarial attacks simultaneously. Specifically, our PUD has a progressive model purification scheme to jointly erase backdoors and enhance the model's adversarial robustness. At the early stage, the adversarial examples of infected models are utilized to erase backdoors. With the backdoor gradually erased, our model purification can naturally turn into a stage to boost the model's robustness against adversarial attacks. Besides, our PUD algorithm can effectively identify poisoned images, which allows the initial extra dataset not to be completely clean. Extensive experimental results show that, our discovered connection between backdoor and adversarial attacks is ubiquitous, no matter what type of backdoor attack. The proposed PUD outperforms the state-of-the-art backdoor defense, including the model repairing-based and data filtering-based methods. Besides, it also has the ability to compete with the most advanced adversarial defense methods.



## **23. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.17894v1) [paper-pdf](http://arxiv.org/pdf/2405.17894v1)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.



## **24. ALA: Naturalness-aware Adversarial Lightness Attack**

cs.CV

9 pages

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2201.06070v3) [paper-pdf](http://arxiv.org/pdf/2201.06070v3)

**Authors**: Yihao Huang, Liangru Sun, Qing Guo, Felix Juefei-Xu, Jiayi Zhu, Jincao Feng, Yang Liu, Geguang Pu

**Abstract**: Most researchers have tried to enhance the robustness of DNNs by revealing and repairing the vulnerability of DNNs with specialized adversarial examples. Parts of the attack examples have imperceptible perturbations restricted by Lp norm. However, due to their high-frequency property, the adversarial examples can be defended by denoising methods and are hard to realize in the physical world. To avoid the defects, some works have proposed unrestricted attacks to gain better robustness and practicality. It is disappointing that these examples usually look unnatural and can alert the guards. In this paper, we propose Adversarial Lightness Attack (ALA), a white-box unrestricted adversarial attack that focuses on modifying the lightness of the images. The shape and color of the samples, which are crucial to human perception, are barely influenced. To obtain adversarial examples with a high attack success rate, we propose unconstrained enhancement in terms of the light and shade relationship in images. To enhance the naturalness of images, we craft the naturalness-aware regularization according to the range and distribution of light. The effectiveness of ALA is verified on two popular datasets for different tasks (i.e., ImageNet for image classification and Places-365 for scene recognition).



## **25. Unmasking Vulnerabilities: Cardinality Sketches under Adaptive Inputs**

cs.DS

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.17780v1) [paper-pdf](http://arxiv.org/pdf/2405.17780v1)

**Authors**: Sara Ahmadian, Edith Cohen

**Abstract**: Cardinality sketches are popular data structures that enhance the efficiency of working with large data sets. The sketches are randomized representations of sets that are only of logarithmic size but can support set merges and approximate cardinality (i.e., distinct count) queries. When queries are not adaptive, that is, they do not depend on preceding query responses, the design provides strong guarantees of correctly answering a number of queries exponential in the sketch size $k$.   In this work, we investigate the performance of cardinality sketches in adaptive settings and unveil inherent vulnerabilities. We design an attack against the ``standard'' estimators that constructs an adversarial input by post-processing responses to a set of simple non-adaptive queries of size linear in the sketch size $k$. Empirically, our attack used only $4k$ queries with the widely used HyperLogLog (HLL++)~\citep{hyperloglog:2007,hyperloglogpractice:EDBT2013} sketch. The simple attack technique suggests it can be effective with post-processed natural workloads. Finally and importantly, we demonstrate that the vulnerability is inherent as \emph{any} estimator applied to known sketch structures can be attacked using a number of queries that is quadratic in $k$, matching a generic upper bound.



## **26. Adversarial Attacks on Hidden Tasks in Multi-Task Learning**

cs.LG

14 pages, 6 figures

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.15244v2) [paper-pdf](http://arxiv.org/pdf/2405.15244v2)

**Authors**: Yu Zhe, Rei Nagaike, Daiki Nishiyama, Kazuto Fukuchi, Jun Sakuma

**Abstract**: Deep learning models are susceptible to adversarial attacks, where slight perturbations to input data lead to misclassification. Adversarial attacks become increasingly effective with access to information about the targeted classifier. In the context of multi-task learning, where a single model learns multiple tasks simultaneously, attackers may aim to exploit vulnerabilities in specific tasks with limited information. This paper investigates the feasibility of attacking hidden tasks within multi-task classifiers, where model access regarding the hidden target task and labeled data for the hidden target task are not available, but model access regarding the non-target tasks is available. We propose a novel adversarial attack method that leverages knowledge from non-target tasks and the shared backbone network of the multi-task model to force the model to forget knowledge related to the target task. Experimental results on CelebA and DeepFashion datasets demonstrate the effectiveness of our method in degrading the accuracy of hidden tasks while preserving the performance of visible tasks, contributing to the understanding of adversarial vulnerabilities in multi-task classifiers.



## **27. Cutting through buggy adversarial example defenses: fixing 1 line of code breaks Sabre**

cs.CR

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.03672v2) [paper-pdf](http://arxiv.org/pdf/2405.03672v2)

**Authors**: Nicholas Carlini

**Abstract**: Sabre is a defense to adversarial examples that was accepted at IEEE S&P 2024. We first reveal significant flaws in the evaluation that point to clear signs of gradient masking. We then show the cause of this gradient masking: a bug in the original evaluation code. By fixing a single line of code in the original repository, we reduce Sabre's robust accuracy to 0%. In response to this, the authors modify the defense and introduce a new defense component not described in the original paper. But this fix contains a second bug; modifying one more line of code reduces robust accuracy to below baseline levels. After we released the first version of our paper online, the authors introduced another change to the defense; by commenting out one line of code during attack we reduce the robust accuracy to 0% again.



## **28. Universal Adversarial Defense in Remote Sensing Based on Pre-trained Denoising Diffusion Models**

cs.CV

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2307.16865v3) [paper-pdf](http://arxiv.org/pdf/2307.16865v3)

**Authors**: Weikang Yu, Yonghao Xu, Pedram Ghamisi

**Abstract**: Deep neural networks (DNNs) have risen to prominence as key solutions in numerous AI applications for earth observation (AI4EO). However, their susceptibility to adversarial examples poses a critical challenge, compromising the reliability of AI4EO algorithms. This paper presents a novel Universal Adversarial Defense approach in Remote Sensing Imagery (UAD-RS), leveraging pre-trained diffusion models to protect DNNs against universal adversarial examples exhibiting heterogeneous patterns. Specifically, a universal adversarial purification framework is developed utilizing pre-trained diffusion models to mitigate adversarial perturbations through the introduction of Gaussian noise and subsequent purification of the perturbations from adversarial examples. Additionally, an Adaptive Noise Level Selection (ANLS) mechanism is introduced to determine the optimal noise level for the purification framework with a task-guided Frechet Inception Distance (FID) ranking strategy, thereby enhancing purification performance. Consequently, only a single pre-trained diffusion model is required for purifying universal adversarial samples with heterogeneous patterns across each dataset, significantly reducing training efforts for multiple attack settings while maintaining high performance without prior knowledge of adversarial perturbations. Experimental results on four heterogeneous RS datasets, focusing on scene classification and semantic segmentation, demonstrate that UAD-RS outperforms state-of-the-art adversarial purification approaches, providing universal defense against seven commonly encountered adversarial perturbations. Codes and the pre-trained models are available online (https://github.com/EricYu97/UAD-RS).



## **29. Spectral regularization for adversarially-robust representation learning**

cs.LG

15 + 15 pages, 8 + 11 figures

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.17181v1) [paper-pdf](http://arxiv.org/pdf/2405.17181v1)

**Authors**: Sheng Yang, Jacob A. Zavatone-Veth, Cengiz Pehlevan

**Abstract**: The vulnerability of neural network classifiers to adversarial attacks is a major obstacle to their deployment in safety-critical applications. Regularization of network parameters during training can be used to improve adversarial robustness and generalization performance. Usually, the network is regularized end-to-end, with parameters at all layers affected by regularization. However, in settings where learning representations is key, such as self-supervised learning (SSL), layers after the feature representation will be discarded when performing inference. For these models, regularizing up to the feature space is more suitable. To this end, we propose a new spectral regularizer for representation learning that encourages black-box adversarial robustness in downstream classification tasks. In supervised classification settings, we show empirically that this method is more effective in boosting test accuracy and robustness than previously-proposed methods that regularize all layers of the network. We then show that this method improves the adversarial robustness of classifiers using representations learned with self-supervised training or transferred from another classification task. In all, our work begins to unveil how representational structure affects adversarial robustness.



## **30. DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation**

cs.CV

* denotes equal contributions; ^ denotes corresponding author. In  this updated version, we have expanded our research to include more  experiments on various adversarial attack methods and latest dataset  distillation studies. All new results have been incorporated into the  document

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2403.13322v2) [paper-pdf](http://arxiv.org/pdf/2403.13322v2)

**Authors**: Yifan Wu, Jiawei Du, Ping Liu, Yuewei Lin, Wenqing Cheng, Wei Xu

**Abstract**: Dataset distillation is an advanced technique aimed at compressing datasets into significantly smaller counterparts, while preserving formidable training performance. Significant efforts have been devoted to promote evaluation accuracy under limited compression ratio while overlooked the robustness of distilled dataset. In this work, we introduce a comprehensive benchmark that, to the best of our knowledge, is the most extensive to date for evaluating the adversarial robustness of distilled datasets in a unified way. Our benchmark significantly expands upon prior efforts by incorporating a wider range of dataset distillation methods, including the latest advancements such as TESLA and SRe2L, a diverse array of adversarial attack methods, and evaluations across a broader and more extensive collection of datasets such as ImageNet-1K. Moreover, we assessed the robustness of these distilled datasets against representative adversarial attack algorithms like PGD and AutoAttack, while exploring their resilience from a frequency perspective. We also discovered that incorporating distilled data into the training batches of the original dataset can yield to improvement of robustness.



## **31. Local Model Reconstruction Attacks in Federated Learning and their Uses**

cs.LG

we discover bugs in experiments

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2210.16205v3) [paper-pdf](http://arxiv.org/pdf/2210.16205v3)

**Authors**: Ilias Driouich, Chuan Xu, Giovanni Neglia, Frederic Giroire, Eoin Thomas

**Abstract**: In this paper, we initiate the study of local model reconstruction attacks for federated learning, where a honest-but-curious adversary eavesdrops the messages exchanged between a targeted client and the server, and then reconstructs the local/personalized model of the victim. The local model reconstruction attack allows the adversary to trigger other classical attacks in a more effective way, since the local model only depends on the client's data and can leak more private information than the global model learned by the server. Additionally, we propose a novel model-based attribute inference attack in federated learning leveraging the local model reconstruction attack. We provide an analytical lower-bound for this attribute inference attack. Empirical results using real world datasets confirm that our local reconstruction attack works well for both regression and classification tasks. Moreover, we benchmark our novel attribute inference attack against the state-of-the-art attacks in federated learning. Our attack results in higher reconstruction accuracy especially when the clients' datasets are heterogeneous. Our work provides a new angle for designing powerful and explainable attacks to effectively quantify the privacy risk in FL.



## **32. Verifying Properties of Binary Neural Networks Using Sparse Polynomial Optimization**

cs.LG

22 pages, 2 figures, 7 tables

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.17049v1) [paper-pdf](http://arxiv.org/pdf/2405.17049v1)

**Authors**: Jianting Yang, Srećko Ðurašinović, Jean-Bernard Lasserre, Victor Magron, Jun Zhao

**Abstract**: This paper explores methods for verifying the properties of Binary Neural Networks (BNNs), focusing on robustness against adversarial attacks. Despite their lower computational and memory needs, BNNs, like their full-precision counterparts, are also sensitive to input perturbations. Established methods for solving this problem are predominantly based on Satisfiability Modulo Theories and Mixed-Integer Linear Programming techniques, which are characterized by NP complexity and often face scalability issues.   We introduce an alternative approach using Semidefinite Programming relaxations derived from sparse Polynomial Optimization. Our approach, compatible with continuous input space, not only mitigates numerical issues associated with floating-point calculations but also enhances verification scalability through the strategic use of tighter first-order semidefinite relaxations. We demonstrate the effectiveness of our method in verifying robustness against both $\|.\|_\infty$ and $\|.\|_2$-based adversarial attacks.



## **33. OSLO: One-Shot Label-Only Membership Inference Attacks**

cs.LG

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.16978v1) [paper-pdf](http://arxiv.org/pdf/2405.16978v1)

**Authors**: Yuefeng Peng, Jaechul Roh, Subhransu Maji, Amir Houmansadr

**Abstract**: We introduce One-Shot Label-Only (OSLO) membership inference attacks (MIAs), which accurately infer a given sample's membership in a target model's training set with high precision using just \emph{a single query}, where the target model only returns the predicted hard label. This is in contrast to state-of-the-art label-only attacks which require $\sim6000$ queries, yet get attack precisions lower than OSLO's. OSLO leverages transfer-based black-box adversarial attacks. The core idea is that a member sample exhibits more resistance to adversarial perturbations than a non-member. We compare OSLO against state-of-the-art label-only attacks and demonstrate that, despite requiring only one query, our method significantly outperforms previous attacks in terms of precision and true positive rate (TPR) under the same false positive rates (FPR). For example, compared to previous label-only MIAs, OSLO achieves a TPR that is 7$\times$ to 28$\times$ stronger under a 0.1\% FPR on CIFAR10 for a ResNet model. We evaluated multiple defense mechanisms against OSLO.



## **34. Adversarial Attacks on Both Face Recognition and Face Anti-spoofing Models**

cs.CV

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.16940v1) [paper-pdf](http://arxiv.org/pdf/2405.16940v1)

**Authors**: Fengfan Zhou, Qianyu Zhou, Xiangtai Li, Xuequan Lu, Lizhuang Ma, Hefei Ling

**Abstract**: Adversarial attacks on Face Recognition (FR) systems have proven highly effective in compromising pure FR models, yet adversarial examples may be ineffective to the complete FR systems as Face Anti-Spoofing (FAS) models are often incorporated and can detect a significant number of them. To address this under-explored and essential problem, we propose a novel setting of adversarially attacking both FR and FAS models simultaneously, aiming to enhance the practicability of adversarial attacks on FR systems. In particular, we introduce a new attack method, namely Style-aligned Distribution Biasing (SDB), to improve the capacity of black-box attacks on both FR and FAS models. Specifically, our SDB framework consists of three key components. Firstly, to enhance the transferability of FAS models, we design a Distribution-aware Score Biasing module to optimize adversarial face examples away from the distribution of spoof images utilizing scores. Secondly, to mitigate the substantial style differences between live images and adversarial examples initialized with spoof images, we introduce an Instance Style Alignment module that aligns the style of adversarial examples with live images. In addition, to alleviate the conflicts between the gradients of FR and FAS models, we propose a Gradient Consistency Maintenance module to minimize disparities between the gradients using Hessian approximation. Extensive experiments showcase the superiority of our proposed attack method to state-of-the-art adversarial attacks.



## **35. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

cs.LG

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.16918v1) [paper-pdf](http://arxiv.org/pdf/2405.16918v1)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization but is also related to adversarial robustness, since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running it runs into a flat uncanny valley where the label remains flipped. We find this phenomenon across various model architectures and datasets. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, the adversarial examples rarely reach a truly flat region. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.



## **36. Accelerating Greedy Coordinate Gradient via Probe Sampling**

cs.CL

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2403.01251v2) [paper-pdf](http://arxiv.org/pdf/2403.01251v2)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a critical issue given their rapid progresses. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing adversarial prompts to break the aligned LLMs, but optimization of GCG is time-consuming. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b-chat and leads to equal or improved attack success rate (ASR) on the AdvBench. Furthermore, probe sampling is also able to accelerate other prompt optimization techniques and adversarial methods, leading to acceleration of $1.8\times$ for AutoPrompt, $2.4\times$ for APE and $2.4\times$ for AutoDAN.



## **37. Rethinking Independent Cross-Entropy Loss For Graph-Structured Data**

cs.LG

20 pages, 4 figures

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.15564v2) [paper-pdf](http://arxiv.org/pdf/2405.15564v2)

**Authors**: Rui Miao, Kaixiong Zhou, Yili Wang, Ninghao Liu, Ying Wang, Xin Wang

**Abstract**: Graph neural networks (GNNs) have exhibited prominent performance in learning graph-structured data. Considering node classification task, based on the i.i.d assumption among node labels, the traditional supervised learning simply sums up cross-entropy losses of the independent training nodes and applies the average loss to optimize GNNs' weights. But different from other data formats, the nodes are naturally connected. It is found that the independent distribution modeling of node labels restricts GNNs' capability to generalize over the entire graph and defend adversarial attacks. In this work, we propose a new framework, termed joint-cluster supervised learning, to model the joint distribution of each node with its corresponding cluster. We learn the joint distribution of node and cluster labels conditioned on their representations, and train GNNs with the obtained joint loss. In this way, the data-label reference signals extracted from the local cluster explicitly strengthen the discrimination ability on the target node. The extensive experiments demonstrate that our joint-cluster supervised learning can effectively bolster GNNs' node classification accuracy. Furthermore, being benefited from the reference signals which may be free from spiteful interference, our learning paradigm significantly protects the node classification from being affected by the adversarial attack.



## **38. Adaptive Batch Normalization Networks for Adversarial Robustness**

cs.LG

Accepted at IEEE International Conference on Advanced Video and  Signal-based Surveillance (AVSS) 2024

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.11708v2) [paper-pdf](http://arxiv.org/pdf/2405.11708v2)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstract**: Deep networks are vulnerable to adversarial examples. Adversarial Training (AT) has been a standard foundation of modern adversarial defense approaches due to its remarkable effectiveness. However, AT is extremely time-consuming, refraining it from wide deployment in practical applications. In this paper, we aim at a non-AT defense: How to design a defense method that gets rid of AT but is still robust against strong adversarial attacks? To answer this question, we resort to adaptive Batch Normalization (BN), inspired by the recent advances in test-time domain adaptation. We propose a novel defense accordingly, referred to as the Adaptive Batch Normalization Network (ABNN). ABNN employs a pre-trained substitute model to generate clean BN statistics and sends them to the target model. The target model is exclusively trained on clean data and learns to align the substitute model's BN statistics. Experimental results show that ABNN consistently improves adversarial robustness against both digital and physically realizable attacks on both image and video datasets. Furthermore, ABNN can achieve higher clean data performance and significantly lower training time complexity compared to AT-based approaches.



## **39. Tradeoffs Between Alignment and Helpfulness in Language Models with Representation Engineering**

cs.CL

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2401.16332v3) [paper-pdf](http://arxiv.org/pdf/2401.16332v3)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. First, we find that under the conditions of our framework, alignment can be guaranteed with representation engineering, and at the same time that helpfulness is harmed in the process. Second, we show that helpfulness is harmed quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.



## **40. Pruning for Robust Concept Erasing in Diffusion Models**

cs.CV

Under review

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2405.16534v1) [paper-pdf](http://arxiv.org/pdf/2405.16534v1)

**Authors**: Tianyun Yang, Juan Cao, Chang Xu

**Abstract**: Despite the impressive capabilities of generating images, text-to-image diffusion models are susceptible to producing undesirable outputs such as NSFW content and copyrighted artworks. To address this issue, recent studies have focused on fine-tuning model parameters to erase problematic concepts. However, existing methods exhibit a major flaw in robustness, as fine-tuned models often reproduce the undesirable outputs when faced with cleverly crafted prompts. This reveals a fundamental limitation in the current approaches and may raise risks for the deployment of diffusion models in the open world. To address this gap, we locate the concept-correlated neurons and find that these neurons show high sensitivity to adversarial prompts, thus could be deactivated when erasing and reactivated again under attacks. To improve the robustness, we introduce a new pruning-based strategy for concept erasing. Our method selectively prunes critical parameters associated with the concepts targeted for removal, thereby reducing the sensitivity of concept-related neurons. Our method can be easily integrated with existing concept-erasing techniques, offering a robust improvement against adversarial inputs. Experimental results show a significant enhancement in our model's ability to resist adversarial inputs, achieving nearly a 40% improvement in erasing the NSFW content and a 30% improvement in erasing artwork style.



## **41. FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering**

cs.CR

20 pages (including bibliography and Appendix), Submitted to ACM CCS  '24

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2310.16152v2) [paper-pdf](http://arxiv.org/pdf/2310.16152v2)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) has become a key component in various language modeling applications such as machine translation, next-word prediction, and medical record analysis. These applications are trained on datasets from many FL participants that often include privacy-sensitive data, such as healthcare records, phone/credit card numbers, login credentials, etc. Although FL enables computation without necessitating clients to share their raw data, determining the extent of privacy leakage in federated language models is challenging and not straightforward. Moreover, existing attacks aim to extract data regardless of how sensitive or naive it is. To fill this research gap, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated large language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other users in FL even without any cooperation from the server. Our best-performing method improves the membership inference recall by 29% and achieves up to 71% private data reconstruction, evidently outperforming existing attacks with stronger assumptions of adversary capabilities.



## **42. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

cs.LG

29 pages

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2405.16405v1) [paper-pdf](http://arxiv.org/pdf/2405.16405v1)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.



## **43. R.A.C.E.: Robust Adversarial Concept Erasure for Secure Text-to-Image Diffusion Model**

cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16341v1) [paper-pdf](http://arxiv.org/pdf/2405.16341v1)

**Authors**: Changhoon Kim, Kyle Min, Yezhou Yang

**Abstract**: In the evolving landscape of text-to-image (T2I) diffusion models, the remarkable capability to generate high-quality images from textual descriptions faces challenges with the potential misuse of reproducing sensitive content. To address this critical issue, we introduce Robust Adversarial Concept Erase (RACE), a novel approach designed to mitigate these risks by enhancing the robustness of concept erasure method for T2I models. RACE utilizes a sophisticated adversarial training framework to identify and mitigate adversarial text embeddings, significantly reducing the Attack Success Rate (ASR). Impressively, RACE achieves a 30 percentage point reduction in ASR for the ``nudity'' concept against the leading white-box attack method. Our extensive evaluations demonstrate RACE's effectiveness in defending against both white-box and black-box attacks, marking a significant advancement in protecting T2I diffusion models from generating inappropriate or misleading imagery. This work underlines the essential need for proactive defense measures in adapting to the rapidly advancing field of adversarial challenges.



## **44. Secure Hierarchical Federated Learning in Vehicular Networks Using Dynamic Client Selection and Anomaly Detection**

cs.LG

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.17497v1) [paper-pdf](http://arxiv.org/pdf/2405.17497v1)

**Authors**: M. Saeid HaghighiFard, Sinem Coleri

**Abstract**: Hierarchical Federated Learning (HFL) faces the significant challenge of adversarial or unreliable vehicles in vehicular networks, which can compromise the model's integrity through misleading updates. Addressing this, our study introduces a novel framework that integrates dynamic vehicle selection and robust anomaly detection mechanisms, aiming to optimize participant selection and mitigate risks associated with malicious contributions. Our approach involves a comprehensive vehicle reliability assessment, considering historical accuracy, contribution frequency, and anomaly records. An anomaly detection algorithm is utilized to identify anomalous behavior by analyzing the cosine similarity of local or model parameters during the federated learning (FL) process. These anomaly records are then registered and combined with past performance for accuracy and contribution frequency to identify the most suitable vehicles for each learning round. Dynamic client selection and anomaly detection algorithms are deployed at different levels, including cluster heads (CHs), cluster members (CMs), and the Evolving Packet Core (EPC), to detect and filter out spurious updates. Through simulation-based performance evaluation, our proposed algorithm demonstrates remarkable resilience even under intense attack conditions. Even in the worst-case scenarios, it achieves convergence times at $63$\% as effective as those in scenarios without any attacks. Conversely, in scenarios without utilizing our proposed algorithm, there is a high likelihood of non-convergence in the FL process.



## **45. Layer-Aware Analysis of Catastrophic Overfitting: Revealing the Pseudo-Robust Shortcut Dependency**

cs.LG

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16262v1) [paper-pdf](http://arxiv.org/pdf/2405.16262v1)

**Authors**: Runqi Lin, Chaojian Yu, Bo Han, Hang Su, Tongliang Liu

**Abstract**: Catastrophic overfitting (CO) presents a significant challenge in single-step adversarial training (AT), manifesting as highly distorted deep neural networks (DNNs) that are vulnerable to multi-step adversarial attacks. However, the underlying factors that lead to the distortion of decision boundaries remain unclear. In this work, we delve into the specific changes within different DNN layers and discover that during CO, the former layers are more susceptible, experiencing earlier and greater distortion, while the latter layers show relative insensitivity. Our analysis further reveals that this increased sensitivity in former layers stems from the formation of pseudo-robust shortcuts, which alone can impeccably defend against single-step adversarial attacks but bypass genuine-robust learning, resulting in distorted decision boundaries. Eliminating these shortcuts can partially restore robustness in DNNs from the CO state, thereby verifying that dependence on them triggers the occurrence of CO. This understanding motivates us to implement adaptive weight perturbations across different layers to hinder the generation of pseudo-robust shortcuts, consequently mitigating CO. Extensive experiments demonstrate that our proposed method, Layer-Aware Adversarial Weight Perturbation (LAP), can effectively prevent CO and further enhance robustness.



## **46. Detecting Adversarial Data via Perturbation Forgery**

cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16226v1) [paper-pdf](http://arxiv.org/pdf/2405.16226v1)

**Authors**: Qian Wang, Chen Li, Yuchen Luo, Hefei Ling, Ping Li, Jiazhong Chen, Shijuan Huang, Ning Yu

**Abstract**: As a defense strategy against adversarial attacks, adversarial detection aims to identify and filter out adversarial data from the data flow based on discrepancies in distribution and noise patterns between natural and adversarial data. Although previous detection methods achieve high performance in detecting gradient-based adversarial attacks, new attacks based on generative models with imbalanced and anisotropic noise patterns evade detection. Even worse, existing techniques either necessitate access to attack data before deploying a defense or incur a significant time cost for inference, rendering them impractical for defending against newly emerging attacks that are unseen by defenders. In this paper, we explore the proximity relationship between adversarial noise distributions and demonstrate the existence of an open covering for them. By learning to distinguish this open covering from the distribution of natural data, we can develop a detector with strong generalization capabilities against all types of adversarial attacks. Based on this insight, we heuristically propose Perturbation Forgery, which includes noise distribution perturbation, sparse mask generation, and pseudo-adversarial data production, to train an adversarial detector capable of detecting unseen gradient-based, generative-model-based, and physical adversarial attacks, while remaining agnostic to any specific models. Comprehensive experiments conducted on multiple general and facial datasets, with a wide spectrum of attacks, validate the strong generalization of our method.



## **47. Enhancing Adversarial Transferability Through Neighborhood Conditional Sampling**

cs.CV

Under review

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16181v1) [paper-pdf](http://arxiv.org/pdf/2405.16181v1)

**Authors**: Chunlin Qiu, Yiheng Duan, Lingchen Zhao, Qian Wang

**Abstract**: Transfer-based attacks craft adversarial examples utilizing a white-box surrogate model to compromise various black-box target models, posing significant threats to many real-world applications. However, existing transfer attacks suffer from either weak transferability or expensive computation. To bridge the gap, we propose a novel sample-based attack, named neighborhood conditional sampling (NCS), which enjoys high transferability with lightweight computation. Inspired by the observation that flat maxima result in better transferability, NCS is formulated as a max-min bi-level optimization problem to seek adversarial regions with high expected adversarial loss and small standard deviations. Specifically, due to the inner minimization problem being computationally intensive to resolve, and affecting the overall transferability, we propose a momentum-based previous gradient inversion approximation (PGIA) method to effectively solve the inner problem without any computation cost. In addition, we prove that two newly proposed attacks, which achieve flat maxima for better transferability, are actually specific cases of NCS under particular conditions. Extensive experiments demonstrate that NCS efficiently generates highly transferable adversarial examples, surpassing the current best method in transferability while requiring only 50% of the computational cost. Additionally, NCS can be seamlessly integrated with other methods to further enhance transferability.



## **48. Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack**

cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16134v1) [paper-pdf](http://arxiv.org/pdf/2405.16134v1)

**Authors**: Mingli Zhu, Siyuan Liang, Baoyuan Wu

**Abstract**: Deep neural networks face persistent challenges in defending against backdoor attacks, leading to an ongoing battle between attacks and defenses. While existing backdoor defense strategies have shown promising performance on reducing attack success rates, can we confidently claim that the backdoor threat has truly been eliminated from the model? To address it, we re-investigate the characteristics of the backdoored models after defense (denoted as defense models). Surprisingly, we find that the original backdoors still exist in defense models derived from existing post-training defense strategies, and the backdoor existence is measured by a novel metric called backdoor existence coefficient. It implies that the backdoors just lie dormant rather than being eliminated. To further verify this finding, we empirically show that these dormant backdoors can be easily re-activated during inference, by manipulating the original trigger with well-designed tiny perturbation using universal adversarial attack. More practically, we extend our backdoor reactivation to black-box scenario, where the defense model can only be queried by the adversary during inference, and develop two effective methods, i.e., query-based and transfer-based backdoor re-activation attacks. The effectiveness of the proposed methods are verified on both image classification and multimodal contrastive learning (i.e., CLIP) tasks. In conclusion, this work uncovers a critical vulnerability that has never been explored in existing defense strategies, emphasizing the urgency of designing more robust and advanced backdoor defense mechanisms in the future.



## **49. Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor**

cs.CR

13 pages, 5 figures and 5 tables

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16112v1) [paper-pdf](http://arxiv.org/pdf/2405.16112v1)

**Authors**: Shaokui Wei, Hongyuan Zha, Baoyuan Wu

**Abstract**: Data-poisoning backdoor attacks are serious security threats to machine learning models, where an adversary can manipulate the training dataset to inject backdoors into models. In this paper, we focus on in-training backdoor defense, aiming to train a clean model even when the dataset may be potentially poisoned. Unlike most existing methods that primarily detect and remove/unlearn suspicious samples to mitigate malicious backdoor attacks, we propose a novel defense approach called PDB (Proactive Defensive Backdoor). Specifically, PDB leverages the "home field" advantage of defenders by proactively injecting a defensive backdoor into the model during training. Taking advantage of controlling the training process, the defensive backdoor is designed to suppress the malicious backdoor effectively while remaining secret to attackers. In addition, we introduce a reversible mapping to determine the defensive target label. During inference, PDB embeds a defensive trigger in the inputs and reverses the model's prediction, suppressing malicious backdoor and ensuring the model's utility on the original task. Experimental results across various datasets and models demonstrate that our approach achieves state-of-the-art defense performance against a wide range of backdoor attacks.



## **50. Mitigating Dialogue Hallucination for Large Vision Language Models via Adversarial Instruction Tuning**

cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2403.10492v2) [paper-pdf](http://arxiv.org/pdf/2403.10492v2)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Vision Language Models,(LVLMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LVLMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues powered by our novel Adversarial Question Generator (AQG), which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LVLMs. On our benchmark, the zero-shot performance of state-of-the-art LVLMs drops significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning (AIT) that robustly fine-tunes LVLMs against hallucinatory dialogues. Extensive experiments show our proposed approach successfully reduces dialogue hallucination while maintaining performance.



