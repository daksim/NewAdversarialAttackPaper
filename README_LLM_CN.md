# Latest Large Language Model Attack Papers
**update at 2024-06-24 09:19:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Safely Learning with Private Data: A Federated Learning Framework for Large Language Model**

利用私人数据安全学习：大型语言模型的联邦学习框架 cs.CR

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.14898v1) [paper-pdf](http://arxiv.org/pdf/2406.14898v1)

**Authors**: JiaYing Zheng, HaiNan Zhang, LingXiang Wang, WangJie Qiu, HongWei Zheng, ZhiMing Zheng

**Abstract**: Private data, being larger and quality-higher than public data, can greatly improve large language models (LLM). However, due to privacy concerns, this data is often dispersed in multiple silos, making its secure utilization for LLM training a challenge. Federated learning (FL) is an ideal solution for training models with distributed private data, but traditional frameworks like FedAvg are unsuitable for LLM due to their high computational demands on clients. An alternative, split learning, offloads most training parameters to the server while training embedding and output layers locally, making it more suitable for LLM. Nonetheless, it faces significant challenges in security and efficiency. Firstly, the gradients of embeddings are prone to attacks, leading to potential reverse engineering of private data. Furthermore, the server's limitation of handle only one client's training request at a time hinders parallel training, severely impacting training efficiency. In this paper, we propose a Federated Learning framework for LLM, named FL-GLM, which prevents data leakage caused by both server-side and peer-client attacks while improving training efficiency. Specifically, we first place the input block and output block on local client to prevent embedding gradient attacks from server. Secondly, we employ key-encryption during client-server communication to prevent reverse engineering attacks from peer-clients. Lastly, we employ optimization methods like client-batching or server-hierarchical, adopting different acceleration methods based on the actual computational capabilities of the server. Experimental results on NLU and generation tasks demonstrate that FL-GLM achieves comparable metrics to centralized chatGLM model, validating the effectiveness of our federated learning framework.

摘要: 私有数据比公共数据更大、质量更高，可以极大地改进大型语言模型(LLM)。然而，出于隐私方面的考虑，这些数据通常分散在多个竖井中，这使得将其安全地用于LLM培训成为一项挑战。联邦学习(FL)是一种适用于具有分布式私有数据的模型训练的理想解决方案，但FedAvg等传统框架由于对客户端的计算要求较高而不适用于LLM。另一种选择是分离学习，将大部分训练参数卸载到服务器，同时在本地训练嵌入和输出层，使其更适合LLM。尽管如此，它在安全和效率方面仍面临重大挑战。首先，嵌入的梯度容易受到攻击，从而导致对私有数据的潜在逆向工程。此外，服务器一次只能处理一个客户端的训练请求的限制阻碍了并行训练，严重影响了训练效率。本文提出了一种用于LLM的联邦学习框架FL-GLM，该框架在提高训练效率的同时，防止了服务器端攻击和对等客户端攻击引起的数据泄漏。具体地说，我们首先将输入块和输出块放置在本地客户端，以防止来自服务器的嵌入梯度攻击。其次，我们在客户-服务器通信过程中使用密钥加密，以防止来自对等客户端的反向工程攻击。最后，我们采用了客户端批处理或服务器分层等优化方法，根据服务器的实际计算能力采用不同的加速方法。在NLU和生成任务上的实验结果表明，FL-GLM达到了与集中式ChatGLM模型相当的指标，验证了联邦学习框架的有效性。



## **2. From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking**

从LLM到MLLM：探索多模式越狱的格局 cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.14859v1) [paper-pdf](http://arxiv.org/pdf/2406.14859v1)

**Authors**: Siyuan Wang, Zhuohan Long, Zhihao Fan, Zhongyu Wei

**Abstract**: The rapid development of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has exposed vulnerabilities to various adversarial attacks. This paper provides a comprehensive overview of jailbreaking research targeting both LLMs and MLLMs, highlighting recent advancements in evaluation benchmarks, attack techniques and defense strategies. Compared to the more advanced state of unimodal jailbreaking, multimodal domain remains underexplored. We summarize the limitations and potential research directions of multimodal jailbreaking, aiming to inspire future research and further enhance the robustness and security of MLLMs.

摘要: 大型语言模型（LLM）和多模式大型语言模型（MLLM）的快速发展暴露了各种对抗攻击的脆弱性。本文全面概述了针对LLM和MLLM的越狱研究，重点介绍了评估基准、攻击技术和防御策略方面的最新进展。与更先进的单模式越狱相比，多模式领域仍然被探索不足。我们总结了多模式越狱的局限性和潜在研究方向，旨在启发未来的研究并进一步增强MLLM的稳健性和安全性。



## **3. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

通过概念激活载体揭示大型语言模型的安全风险 cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2404.12038v2) [paper-pdf](http://arxiv.org/pdf/2404.12038v2)

**Authors**: Zhihao Xu, Ruixuan Huang, Shuai Wang, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, we find that six out of seven open-source LLMs that we attack consistently provide relevant answers to more than 85\% malicious instructions. Finally, we provide insights into the safety mechanism of LLMs.

摘要: 尽管进行了仔细的安全调整，但当前的大型语言模型(LLM)仍然容易受到各种攻击。为了进一步揭示LLMS的安全隐患，我们引入了安全概念激活向量(SCAV)框架，通过准确解释LLMS的安全机制来有效地指导攻击。然后，我们开发了一种SCAV引导的攻击方法，该方法可以生成攻击提示和带有自动选择的扰动超参数的嵌入级攻击。自动和人工评估都表明，我们的攻击方法在需要更少的训练数据的情况下，显著地提高了攻击成功率和响应质量。此外，我们发现我们生成的攻击提示可以转移到GPT-4上，嵌入级攻击也可以转移到参数已知的其他白盒LLM上。我们的实验进一步揭示了当前LLM中存在的安全风险。例如，我们发现，我们攻击的七个开源LLM中有六个始终为超过85%的恶意指令提供相关答案。最后，我们对LLMS的安全机制提供了见解。



## **4. FedSecurity: Benchmarking Attacks and Defenses in Federated Learning and Federated LLMs**

FedSecurity：联邦学习和联邦LLM中的攻击和防御基准 cs.CR

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2306.04959v5) [paper-pdf](http://arxiv.org/pdf/2306.04959v5)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Carlee Joe-Wong, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedSecurity, an end-to-end benchmark that serves as a supplementary component of the FedML library for simulating adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). FedSecurity eliminates the need for implementing the fundamental FL procedures, e.g., FL training and data loading, from scratch, thus enables users to focus on developing their own attack and defense strategies. It contains two key components, including FedAttacker that conducts a variety of attacks during FL training, and FedDefender that implements defensive mechanisms to counteract these attacks. FedSecurity has the following features: i) It offers extensive customization options to accommodate a broad range of machine learning models (e.g., Logistic Regression, ResNet, and GAN) and FL optimizers (e.g., FedAVG, FedOPT, and FedNOVA); ii) it enables exploring the effectiveness of attacks and defenses across different datasets and models; and iii) it supports flexible configuration and customization through a configuration file and some APIs. We further demonstrate FedSecurity's utility and adaptability through federated training of Large Language Models (LLMs) to showcase its potential on a wide range of complex applications.

摘要: 本文介绍了FedSecurity，这是一个端到端的基准测试，作为FedML库的补充组件，用于模拟联邦学习中的对抗性攻击和相应的防御机制。FedSecurity不需要从头开始实施基本的FL程序，例如FL训练和数据加载，从而使用户能够专注于开发他们自己的攻击和防御策略。它包含两个关键组件，包括在FL训练期间进行各种攻击的FedAttacker和实现防御机制以对抗这些攻击的FedDefender。FedSecurity具有以下功能：i)它提供广泛的定制选项，以适应广泛的机器学习模型(例如Logistic回归、ResNet和GAN)和FL优化器(例如FedAVG、FedOPT和FedNOVA)；ii)它能够跨不同的数据集和模型探索攻击和防御的有效性；iii)它通过一个配置文件和一些API支持灵活的配置和定制。通过对大型语言模型(LLM)的联合训练，我们进一步展示了FedSecurity的实用性和适应性，以展示其在广泛的复杂应用中的潜力。



## **5. MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate**

多Agent协作攻击：通过辩论调查大型语言模型协作中的对抗性攻击 cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14711v1) [paper-pdf](http://arxiv.org/pdf/2406.14711v1)

**Authors**: Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, William Wang

**Abstract**: Large Language Models (LLMs) have shown exceptional results on current benchmarks when working individually. The advancement in their capabilities, along with a reduction in parameter size and inference times, has facilitated the use of these models as agents, enabling interactions among multiple models to execute complex tasks. Such collaborations offer several advantages, including the use of specialized models (e.g. coding), improved confidence through multiple computations, and enhanced divergent thinking, leading to more diverse outputs. Thus, the collaborative use of language models is expected to grow significantly in the coming years. In this work, we evaluate the behavior of a network of models collaborating through debate under the influence of an adversary. We introduce pertinent metrics to assess the adversary's effectiveness, focusing on system accuracy and model agreement. Our findings highlight the importance of a model's persuasive ability in influencing others. Additionally, we explore inference-time methods to generate more compelling arguments and evaluate the potential of prompt-based mitigation as a defensive strategy.

摘要: 大型语言模型(LLM)在单独工作时，在当前基准上显示了特殊的结果。它们能力的进步，加上参数大小和推理时间的减少，促进了这些模型作为代理的使用，使多个模型之间能够相互作用，以执行复杂的任务。这种协作提供了几个优势，包括使用专门的模型(例如编码)、通过多次计算提高信心以及增强发散思维，从而产生更多样化的产出。因此，语言模型的协作使用预计在未来几年将显著增长。在这项工作中，我们评估了一个模型网络在对手的影响下通过辩论进行合作的行为。我们引入了相关的度量来评估对手的有效性，重点是系统的准确性和模型的一致性。我们的发现突显了模特的说服力在影响他人方面的重要性。此外，我们探索推理时间方法来生成更令人信服的论点，并评估基于即时缓解作为一种防御策略的潜力。



## **6. Unmasking Database Vulnerabilities: Zero-Knowledge Schema Inference Attacks in Text-to-SQL Systems**

揭露数据库漏洞：文本转SQL系统中的零知识模式推理攻击 cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14545v1) [paper-pdf](http://arxiv.org/pdf/2406.14545v1)

**Authors**: Đorđe Klisura, Anthony Rios

**Abstract**: Relational databases are integral to modern information systems, serving as the foundation for storing, querying, and managing data efficiently and effectively. Advancements in large language modeling have led to the emergence of text-to-SQL technologies, significantly enhancing the querying and extracting of information from these databases and raising concerns about privacy and security. Our research extracts the database schema elements underlying a text-to-SQL model. Knowledge of the schema can make attacks such as SQL injection easier. By asking specially crafted questions, we have developed a zero-knowledge framework designed to probe various database schema elements without knowledge of the database itself. The text-to-SQL models then process these questions to produce an output that we use to uncover the structure of the database schema. We apply it to specialized text-to-SQL models fine-tuned on text-SQL pairs and generative language models used for SQL generation. Overall, we can reconstruct the table names with an F1 of nearly .75 for fine-tuned models and .96 for generative.

摘要: 关系数据库是现代信息系统不可或缺的组成部分，是高效存储、查询和管理数据的基础。大型语言建模的进步导致了文本到SQL技术的出现，极大地增强了从这些数据库查询和提取信息的能力，并引发了对隐私和安全的担忧。我们的研究提取了Text-to-SQL模型下的数据库模式元素。了解模式可以使SQL注入等攻击变得更容易。通过提出精心设计的问题，我们开发了一个零知识框架，旨在探索各种数据库模式元素，而不需要了解数据库本身。然后，Text-to-SQL模型处理这些问题，以产生我们用来揭示数据库模式结构的输出。我们将其应用于专门的文本到SQL模型，这些模型在文本-SQL对和用于生成SQL的生成语言模型上进行了微调。总体而言，我们可以重新构建表名，对于微调模型，F1接近0.75，对于生成性模型，F1接近0.96。



## **7. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14393v1) [paper-pdf](http://arxiv.org/pdf/2406.14393v1)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.

摘要: 大型语言模型（LLM）的广泛采用引发了人们对其安全性和可靠性的担忧，特别是对其容易受到对抗攻击的影响。在本文中，我们提出了一种新颖的视角，将此漏洞归因于对齐过程中的奖励错误指定。我们引入了一个指标ReGap来量化奖励错误指定的程度，并展示其在检测有害后门提示方面的有效性和稳健性。在这些见解的基础上，我们介绍了ReMiss，这是一个用于自动化红色分组的系统，可以针对各种目标对齐的LLM生成对抗提示。ReMiss在AdvBench基准上实现了最先进的攻击成功率，同时保留了生成提示的人类可读性。与以前的方法相比，详细的分析强调了拟议的奖励错误指定目标所带来的独特优势。



## **8. Safety of Multimodal Large Language Models on Images and Texts**

图像和文本上多模式大型语言模型的安全性 cs.CV

Accepted at IJCAI2024

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2402.00357v3) [paper-pdf](http://arxiv.org/pdf/2402.00357v3)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Attracted by the impressive power of Multimodal Large Language Models (MLLMs), the public is increasingly utilizing them to improve the efficiency of daily work. Nonetheless, the vulnerabilities of MLLMs to unsafe instructions bring huge safety risks when these models are deployed in real-world scenarios. In this paper, we systematically survey current efforts on the evaluation, attack, and defense of MLLMs' safety on images and text. We begin with introducing the overview of MLLMs on images and text and understanding of safety, which helps researchers know the detailed scope of our survey. Then, we review the evaluation datasets and metrics for measuring the safety of MLLMs. Next, we comprehensively present attack and defense techniques related to MLLMs' safety. Finally, we analyze several unsolved issues and discuss promising research directions. The latest papers are continually collected at https://github.com/isXinLiu/MLLM-Safety-Collection.

摘要: 受多模式大型语言模型（MLLM）令人印象深刻的力量的吸引，公众越来越多地利用它们来提高日常工作的效率。尽管如此，当这些模型部署在现实世界场景中时，MLLM对不安全指令的脆弱性带来了巨大的安全风险。在本文中，我们系统地调查了当前对MLLM图像和文本安全性的评估、攻击和防御方面的工作。我们首先介绍MLLM关于图像和文本的概述以及对安全性的理解，这有助于研究人员了解我们调查的详细范围。然后，我们审查用于衡量MLLM安全性的评估数据集和指标。接下来，我们全面介绍与MLLM安全相关的攻击和防御技术。最后，我们分析了几个尚未解决的问题并讨论了有前途的研究方向。https://github.com/isXinLiu/MLLM-Safety-Collection不断收集最新论文。



## **9. Are you still on track!? Catching LLM Task Drift with Activations**

你还在正轨上吗！？通过激活捕捉LLM任务漂移 cs.CR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.00799v3) [paper-pdf](http://arxiv.org/pdf/2406.00799v3)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models (LLMs) are routinely used in retrieval-augmented applications to orchestrate tasks and process inputs from users and other sources. These inputs, even in a single LLM interaction, can come from a variety of sources, of varying trustworthiness and provenance. This opens the door to prompt injection attacks, where the LLM receives and acts upon instructions from supposedly data-only sources, thus deviating from the user's original instructions. We define this as task drift, and we propose to catch it by scanning and analyzing the LLM's activations. We compare the LLM's activations before and after processing the external input in order to detect whether this input caused instruction drift. We develop two probing methods and find that simply using a linear classifier can detect drift with near perfect ROC AUC on an out-of-distribution test set. We show that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Our setup does not require any modification of the LLM (e.g., fine-tuning) or any text generation, thus maximizing deployability and cost efficiency and avoiding reliance on unreliable model output. To foster future research on activation-based task inspection, decoding, and interpretability, we will release our large-scale TaskTracker toolkit, comprising a dataset of over 500K instances, representations from 4 SoTA language models, and inspection tools.

摘要: 大型语言模型(LLM)通常用于检索增强的应用程序中，以协调任务并处理来自用户和其他来源的输入。这些输入，即使是在单个LLM交互中，也可以来自各种来源，具有不同的可信度和出处。这为即时注入攻击打开了大门，在这种情况下，LLM接收来自假定仅限数据的来源的指令并对其采取行动，从而偏离用户的原始指令。我们将其定义为任务漂移，并建议通过扫描和分析LLM的激活来捕获它。我们比较LLM在处理外部输入之前和之后的激活，以检测该输入是否导致指令漂移。我们开发了两种探测方法，发现简单地使用线性分类器可以在非分布测试集上以接近完美的ROC AUC来检测漂移。我们表明，这种方法对于看不见的任务领域(如提示注入、越狱和恶意指令)的泛化效果出奇地好，而且没有接受过任何这些攻击的培训。我们的设置不需要对LLM进行任何修改(例如，微调)或任何文本生成，从而最大限度地提高可部署性和成本效益，并避免依赖不可靠的模型输出。为了促进未来对基于激活的任务检测、解码和可解释性的研究，我们将发布我们的大型TaskTracker工具包，其中包括超过50万个实例的数据集、来自4个SOTA语言模型的表示和检测工具。



## **10. FewFedPIT: Towards Privacy-preserving and Few-shot Federated Instruction Tuning**

FewFedPIT：迈向隐私保护和少镜头联邦指令调优 cs.CR

Work in progress

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2403.06131v2) [paper-pdf](http://arxiv.org/pdf/2403.06131v2)

**Authors**: Zhuo Zhang, Jingyuan Zhang, Jintao Huang, Lizhen Qu, Hongzhi Zhang, Qifan Wang, Xun Zhou, Zenglin Xu

**Abstract**: Instruction tuning has been identified as a crucial technique for optimizing the performance of large language models (LLMs) in generating human-aligned responses. Nonetheless, gathering diversified and superior-quality instruction data for such tuning presents notable obstacles, especially in domains with rigid privacy provisions. Federated instruction tuning (FedIT) has emerged as a promising solution, by consolidating collaborative training across multiple data owners, thereby resulting in a privacy-preserving learning model. However, FedIT encounters limitations such as scarcity of instructional data and risk of exposure to training data extraction attacks. In this paper, we propose a novel federated algorithm, FewFedPIT, designed to simultaneously enhance privacy protection and model performance of federated few-shot learning. FewFedPITcomprises three vital components on the client side: (1) synthetic data generation, which utilizes LLMs' in-context learning capacity to generate synthetic data autonomously, thus expanding the local database; (2) parameter isolation training, which individually updates the public parameters in the synthetic data and the private parameters in the local data, consequently mitigating the noise impact of the synthetic data; (3) local aggregation sharing, which mixes public and private parameters before uploading, effectively preventing data extraction attacks. Extensive experiments on three open-source datasets demonstrate the effectiveness of FewFedPITin, enhancing privacy preservation and improving federated few-shot performance.

摘要: 指令调优已被认为是优化大语言模型(LLM)生成人类对齐响应的性能的关键技术。尽管如此，为这种调整收集多样化和高质量的教学数据存在明显的障碍，特别是在隐私条款严格的领域。联邦教学调整(FedIT)通过整合跨多个数据所有者的协作培训，从而产生保护隐私的学习模型，已成为一种有前途的解决方案。然而，FedIT遇到了诸如教学数据稀缺和暴露于训练数据提取攻击的风险等限制。在本文中，我们提出了一种新的联邦算法FewFedPIT，旨在同时增强隐私保护和联邦少镜头学习的模型性能。FewFedPIT在客户端包括三个重要组成部分：(1)合成数据生成，利用LLMS的上下文学习能力自主生成合成数据，从而扩展本地数据库；(2)参数隔离训练，分别更新合成数据中的公共参数和本地数据中的私有参数，从而减轻合成数据的噪声影响；(3)本地聚合共享，在上传之前混合公有和私有参数，有效防止数据提取攻击。在三个开源数据集上的大量实验证明了FewFedPITin的有效性，增强了隐私保护，提高了联邦少镜头性能。



## **11. Protecting Privacy Through Approximating Optimal Parameters for Sequence Unlearning in Language Models**

通过逼近语言模型中序列取消学习的最佳参数来保护隐私 cs.CL

Accepted to ACL2024 findings

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14091v1) [paper-pdf](http://arxiv.org/pdf/2406.14091v1)

**Authors**: Dohyun Lee, Daniel Rim, Minseok Choi, Jaegul Choo

**Abstract**: Although language models (LMs) demonstrate exceptional capabilities on various tasks, they are potentially vulnerable to extraction attacks, which represent a significant privacy risk. To mitigate the privacy concerns of LMs, machine unlearning has emerged as an important research area, which is utilized to induce the LM to selectively forget about some of its training data. While completely retraining the model will guarantee successful unlearning and privacy assurance, it is impractical for LMs, as it would be time-consuming and resource-intensive. Prior works efficiently unlearn the target token sequences, but upon subsequent iterations, the LM displays significant degradation in performance. In this work, we propose Privacy Protection via Optimal Parameters (POP), a novel unlearning method that effectively forgets the target token sequences from the pretrained LM by applying optimal gradient updates to the parameters. Inspired by the gradient derivation of complete retraining, we approximate the optimal training objective that successfully unlearns the target sequence while retaining the knowledge from the rest of the training data. Experimental results demonstrate that POP exhibits remarkable retention performance post-unlearning across 9 classification and 4 dialogue benchmarks, outperforming the state-of-the-art by a large margin. Furthermore, we introduce Remnant Memorization Accuracy that quantifies privacy risks based on token likelihood and validate its effectiveness through both qualitative and quantitative analyses.

摘要: 尽管语言模型(LMS)在各种任务上表现出非凡的能力，但它们可能容易受到提取攻击，这代表着重大的隐私风险。为了缓解LMS的隐私问题，机器遗忘已经成为一个重要的研究领域，它被用来诱导LM选择性地忘记它的一些训练数据。虽然完全再培训该模型将确保成功忘记学习和隐私保证，但这对LMS来说是不切实际的，因为它将耗时和资源密集型。先前的工作有效地取消学习目标令牌序列，但在随后的迭代中，LM表现出显著的性能下降。在这项工作中，我们提出了通过最优参数的隐私保护(POP)，这是一种新的去学习方法，通过对参数应用最优梯度更新来有效地从预先训练的LM中忘记目标令牌序列。受完全再训练的梯度导数的启发，我们逼近了最优训练目标，在保留其余训练数据的知识的同时，成功地去除了目标序列。实验结果表明，在9个分类和4个对话基准中，POP在遗忘后表现出显著的保持性能，远远超过最新水平。此外，我们还引入了基于令牌似然量化隐私风险的剩余记忆准确率，并通过定性和定量分析验证了其有效性。



## **12. Prompt Injection Attacks in Defended Systems**

防御系统中的即时注入攻击 cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14048v1) [paper-pdf](http://arxiv.org/pdf/2406.14048v1)

**Authors**: Daniil Khomsky, Narek Maloyan, Bulat Nutfullin

**Abstract**: Large language models play a crucial role in modern natural language processing technologies. However, their extensive use also introduces potential security risks, such as the possibility of black-box attacks. These attacks can embed hidden malicious features into the model, leading to adverse consequences during its deployment.   This paper investigates methods for black-box attacks on large language models with a three-tiered defense mechanism. It analyzes the challenges and significance of these attacks, highlighting their potential implications for language processing system security. Existing attack and defense methods are examined, evaluating their effectiveness and applicability across various scenarios.   Special attention is given to the detection algorithm for black-box attacks, identifying hazardous vulnerabilities in language models and retrieving sensitive information. This research presents a methodology for vulnerability detection and the development of defensive strategies against black-box attacks on large language models.

摘要: 大型语言模型在现代自然语言处理技术中起着至关重要的作用。然而，它们的广泛使用也带来了潜在的安全风险，例如可能发生黑匣子攻击。这些攻击可能会将隐藏的恶意功能嵌入到模型中，导致部署过程中的不良后果。研究了三层防御机制对大型语言模型进行黑盒攻击的方法。它分析了这些攻击的挑战和意义，强调了它们对语言处理系统安全的潜在影响。检查了现有的攻击和防御方法，评估了它们在各种情况下的有效性和适用性。对黑盒攻击的检测算法、识别语言模型中的危险漏洞和检索敏感信息给予了特别关注。这项研究提出了一种针对大型语言模型的漏洞检测和防御策略的开发方法。



## **13. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

从心理测量学角度通过攻击来评估大型语言模型中的内隐偏差 cs.CL

Code and datasets are available at  https://github.com/wen112358/ImplicitBiasPsychometricEvaluation

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14023v1) [paper-pdf](http://arxiv.org/pdf/2406.14023v1)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As Large Language Models (LLMs) become an important way of information seeking, there have been increasing concerns about the unethical content LLMs may generate. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain groups by attacking them with carefully crafted instructions to elicit biased responses. Our attack methodology is inspired by psychometric principles in cognitive and social psychology. We propose three attack approaches, i.e., Disguise, Deception, and Teaching, based on which we built evaluation datasets for four common bias types. Each prompt attack has bilingual versions. Extensive evaluation of representative LLMs shows that 1) all three attack methods work effectively, especially the Deception attacks; 2) GLM-3 performs the best in defending our attacks, compared to GPT-3.5 and GPT-4; 3) LLMs could output content of other bias types when being taught with one type of bias. Our methodology provides a rigorous and effective way of evaluating LLMs' implicit bias and will benefit the assessments of LLMs' potential ethical risks.

摘要: 随着大型语言模型成为人们寻找信息的一种重要方式，人们越来越关注大型语言模型可能产生的不道德内容。在这篇文章中，我们对LLMS对某些群体的内隐偏见进行了严格的评估，通过精心设计的指令来攻击他们，以获得有偏见的反应。我们的攻击方法受到认知和社会心理学中的心理测量学原理的启发。我们提出了三种攻击方法，即伪装、欺骗和教学，并在此基础上建立了四种常见偏差类型的评估数据集。每个即时攻击都有双语版本。对有代表性的LLMS的广泛评估表明：1)三种攻击方法都有效，尤其是欺骗性攻击；2)GLM-3在防御我们的攻击方面表现最好，相比GPT-3.5和GPT-4；3)LLMS在被教授一种类型的偏向时可以输出其他偏向类型的内容。我们的方法提供了一种严格而有效的方法来评估低收入者的隐性偏见，并将有助于评估低收入者的潜在道德风险。



## **14. Mitigating Fine-tuning based Jailbreak Attack with Backdoor Enhanced Safety Alignment**

通过后门增强的安全调整缓解基于微调的越狱攻击 cs.CR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2402.14968v3) [paper-pdf](http://arxiv.org/pdf/2402.14968v3)

**Authors**: Jiongxiao Wang, Jiazhao Li, Yiquan Li, Xiangyu Qi, Junjie Hu, Yixuan Li, Patrick McDaniel, Muhao Chen, Bo Li, Chaowei Xiao

**Abstract**: Despite the general capabilities of Large Language Models (LLM), these models still request fine-tuning or adaptation with customized data when meeting specific business demands. However, this process inevitably introduces new threats, particularly against the Fine-tuning based Jailbreak Attack (FJAttack) under the setting of Language-Model-as-a-Service (LMaaS), where the model's safety has been significantly compromised by fine-tuning users' uploaded examples contain just a few harmful examples. Though potential defenses have been proposed that the service providers can integrate safety examples into the fine-tuning dataset to reduce safety issues, such approaches require incorporating a substantial amount of data, making it inefficient. To effectively defend against the FJAttack with limited safety examples under LMaaS, we propose the Backdoor Enhanced Safety Alignment method inspired by an analogy with the concept of backdoor attacks. In particular, service providers will construct prefixed safety examples with a secret prompt, acting as a "backdoor trigger". By integrating prefixed safety examples into the fine-tuning dataset, the subsequent fine-tuning process effectively acts as the "backdoor attack", establishing a strong correlation between the secret prompt and safety generations. Consequently, safe responses are ensured once service providers prepend this secret prompt ahead of any user input during inference. Our comprehensive experiments demonstrate that through the Backdoor Enhanced Safety Alignment with adding as few as 11 prefixed safety examples, the maliciously fine-tuned LLMs will achieve similar safety performance as the original aligned models without harming the benign performance. Furthermore, we also present the effectiveness of our method in a more practical setting where the fine-tuning data consists of both FJAttack examples and the fine-tuning task data.

摘要: 尽管大型语言模型(LLM)具有一般功能，但在满足特定业务需求时，这些模型仍然需要使用定制数据进行微调或调整。然而，这一过程不可避免地带来了新的威胁，特别是针对LMaaS(Language-Model-as-a-Service，语言模型即服务)设置下的基于Fine-Tuning的越狱攻击(FJAttack)，其中模型的安全性因微调用户上传的示例仅包含几个有害示例而受到严重威胁。尽管有人提出了潜在的防御措施，即服务提供商可以将安全实例整合到微调数据集中，以减少安全问题，但这种方法需要纳入大量数据，使其效率低下。为了在LMaaS环境下有效防御安全实例有限的FJAttack，我们借鉴了后门攻击的概念，提出了后门增强安全对齐方法。特别是，服务提供商将构建带有前缀的安全示例，并使用秘密提示，充当“后门触发器”。通过将前缀的安全实例整合到微调数据集中，后续的微调过程有效地充当了“后门攻击”，在秘密提示和安全生成之间建立了很强的关联。因此，一旦服务提供商在推理过程中在任何用户输入之前预先考虑此秘密提示，就可以确保安全响应。我们的综合实验表明，通过添加仅需11个前缀安全实例的后门增强安全对准，恶意微调的LLM将在不损害良性性能的情况下获得与原始对准模型相似的安全性能。此外，我们还在一个更实际的环境中展示了我们的方法的有效性，其中微调数据包括FJAttack实例和微调任务数据。



## **15. RLHFPoison: Reward Poisoning Attack for Reinforcement Learning with Human Feedback in Large Language Models**

RL HFPoison：大型语言模型中具有人类反馈的强化学习的奖励中毒攻击 cs.AI

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2311.09641v2) [paper-pdf](http://arxiv.org/pdf/2311.09641v2)

**Authors**: Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a methodology designed to align Large Language Models (LLMs) with human preferences, playing an important role in LLMs alignment. Despite its advantages, RLHF relies on human annotators to rank the text, which can introduce potential security vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the ranking score by up-ranking any malicious text to steer the LLM adversarially. To assess the red-teaming of RLHF against human preference data poisoning, we propose RankPoison, a poisoning attack method on candidates' selection of preference rank flipping to reach certain malicious behaviors (e.g., generating longer sequences, which can increase the computational cost). With poisoned dataset generated by RankPoison, we can perform poisoning attacks on LLMs to generate longer tokens without hurting the original safety alignment performance. Moreover, applying RankPoison, we also successfully implement a backdoor attack where LLMs can generate longer answers under questions with the trigger word. Our findings highlight critical security challenges in RLHF, underscoring the necessity for more robust alignment methods for LLMs.

摘要: 带人反馈的强化学习(RLHF)是一种将大语言模型与人的偏好相匹配的方法，在大语言模型对齐中起着重要作用。尽管RLHF有其优势，但它依靠人工注释者对文本进行排名，如果任何敌意注释者(即攻击者)通过对任何恶意文本进行排名来操纵排名分数，从而对LLM进行敌意操作，这可能会引入潜在的安全漏洞。为了评估RLHF的红团队对抗人类偏好数据中毒的能力，我们提出了一种毒化攻击方法RankPoison，该方法针对候选者选择偏好翻转来达到某些恶意行为(例如，生成更长的序列，这会增加计算成本)。利用RankPoison生成的有毒数据集，我们可以在不损害原始安全对齐性能的情况下，对LLM进行中毒攻击，生成更长的令牌。此外，应用RankPoison，我们还成功地实现了一个后门攻击，在带有触发词的问题下，LLMS可以生成更长的答案。我们的发现突出了RLHF中的关键安全挑战，强调了对LLM采用更强大的比对方法的必要性。



## **16. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

中毒是对LLM联盟的真正威胁吗？也许比你想象的还要多 cs.LG

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.12091v2) [paper-pdf](http://arxiv.org/pdf/2406.12091v2)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.

摘要: 人类反馈强化学习(RLHF)的最新进展对大型语言模型(LLM)的匹配产生了重大影响。强化学习算法的敏感性，如最近策略优化(PPO)，导致了直接策略优化(DPO)的新工作，它在监督学习框架中处理RLHF。这些RLHF方法的实际使用越来越多，因此有理由对其脆弱性进行分析。在这项工作中，我们调查了DPO在不同场景下对中毒攻击的脆弱性，并比较了偏好中毒的有效性，这是第一次。我们全面分析了DPO在不同类型的攻击下的漏洞，即后门攻击和非后门攻击，以及不同的中毒方法，跨越了广泛的语言模型，即：大羊驼7B、米斯特拉尔7B和杰玛7B。我们发现，与基于PPO的方法不同，当涉及到后门攻击时，需要至少4%的数据被毒化才能引发有害行为，而我们更简单地利用DPO的真正漏洞，因此我们只需使用多达0.5%的数据就可以毒害模型。我们进一步调查了该漏洞背后的潜在原因，以及该漏洞在多大程度上转化为后门攻击与非后门攻击。



## **17. ObscurePrompt: Jailbreaking Large Language Models via Obscure Input**

晦涩提示：通过晦涩输入破解大型语言模型 cs.CL

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13662v1) [paper-pdf](http://arxiv.org/pdf/2406.13662v1)

**Authors**: Yue Huang, Jingyu Tang, Dongping Chen, Bingda Tang, Yao Wan, Lichao Sun, Xiangliang Zhang

**Abstract**: Recently, Large Language Models (LLMs) have garnered significant attention for their exceptional natural language processing capabilities. However, concerns about their trustworthiness remain unresolved, particularly in addressing "jailbreaking" attacks on aligned LLMs. Previous research predominantly relies on scenarios with white-box LLMs or specific and fixed prompt templates, which are often impractical and lack broad applicability. In this paper, we introduce a straightforward and novel method, named ObscurePrompt, for jailbreaking LLMs, inspired by the observed fragile alignments in Out-of-Distribution (OOD) data. Specifically, we first formulate the decision boundary in the jailbreaking process and then explore how obscure text affects LLM's ethical decision boundary. ObscurePrompt starts with constructing a base prompt that integrates well-known jailbreaking techniques. Powerful LLMs are then utilized to obscure the original prompt through iterative transformations, aiming to bolster the attack's robustness. Comprehensive experiments show that our approach substantially improves upon previous methods in terms of attack effectiveness, maintaining efficacy against two prevalent defense mechanisms. We believe that our work can offer fresh insights for future research on enhancing LLM alignment.

摘要: 近年来，大型语言模型(LLM)以其卓越的自然语言处理能力引起了人们的极大关注。然而，对它们可信度的担忧仍然没有得到解决，特别是在解决对结盟的LLM的“越狱”攻击方面。以前的研究主要依赖于白盒LLM或特定和固定提示模板的场景，这些场景往往不切实际，缺乏广泛的适用性。在这篇文章中，我们介绍了一个简单而新颖的方法，称为ObscurePrompt，用于越狱LLMS，灵感来自于观察到的分布外(OOD)数据中的脆弱对齐。具体地说，我们首先阐述了越狱过程中的决策边界，然后探讨了晦涩的文本如何影响LLM的伦理决策边界。ObscurePrompt首先构建一个集成了众所周知的越狱技术的基本提示。然后利用强大的LLM通过迭代变换来模糊原始提示，旨在增强攻击的健壮性。综合实验表明，我们的方法在攻击有效性方面比以前的方法有了很大的提高，保持了对两种流行的防御机制的有效性。我们相信，我们的工作可以为未来增强LLM对齐的研究提供新的见解。



## **18. Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation**

撬锁LLM：使用代币级操纵的基于日志的越狱 cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2405.13068v2) [paper-pdf](http://arxiv.org/pdf/2405.13068v2)

**Authors**: Yuxi Li, Yi Liu, Yuekang Li, Ling Shi, Gelei Deng, Shengquan Chen, Kailong Wang

**Abstract**: Large language models (LLMs) have transformed the field of natural language processing, but they remain susceptible to jailbreaking attacks that exploit their capabilities to generate unintended and potentially harmful content. Existing token-level jailbreaking techniques, while effective, face scalability and efficiency challenges, especially as models undergo frequent updates and incorporate advanced defensive measures. In this paper, we introduce JailMine, an innovative token-level manipulation approach that addresses these limitations effectively. JailMine employs an automated "mining" process to elicit malicious responses from LLMs by strategically selecting affirmative outputs and iteratively reducing the likelihood of rejection. Through rigorous testing across multiple well-known LLMs and datasets, we demonstrate JailMine's effectiveness and efficiency, achieving a significant average reduction of 86% in time consumed while maintaining high success rates averaging 95%, even in the face of evolving defensive strategies. Our work contributes to the ongoing effort to assess and mitigate the vulnerability of LLMs to jailbreaking attacks, underscoring the importance of continued vigilance and proactive measures to enhance the security and reliability of these powerful language models.

摘要: 大型语言模型(LLM)已经改变了自然语言处理领域，但它们仍然容易受到越狱攻击，这些攻击利用它们的能力生成意外的和潜在的有害内容。现有的令牌级越狱技术虽然有效，但面临可伸缩性和效率的挑战，特别是在模型经历频繁更新和采用先进防御措施的情况下。在本文中，我们介绍了Jailmy，一种创新的令牌级操作方法，有效地解决了这些限制。Jailmine使用一个自动化的“挖掘”过程，通过战略性地选择肯定的输出并反复降低拒绝的可能性，来引发来自LLMS的恶意响应。通过对多个知名LLM和数据集的严格测试，我们展示了Jailmine的有效性和效率，实现了平均86%的时间消耗显著减少，同时保持了平均95%的高成功率，即使面对不断变化的防御策略。我们的工作有助于评估和减轻LLMS在越狱攻击中的脆弱性，强调了继续保持警惕和采取积极措施以增强这些强大语言模型的安全性和可靠性的重要性。



## **19. Textual Unlearning Gives a False Sense of Unlearning**

文本遗忘给人一种遗忘的错误感觉 cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13348v1) [paper-pdf](http://arxiv.org/pdf/2406.13348v1)

**Authors**: Jiacheng Du, Zhibo Wang, Kui Ren

**Abstract**: Language models (LMs) are susceptible to "memorizing" training data, including a large amount of private or copyright-protected content. To safeguard the right to be forgotten (RTBF), machine unlearning has emerged as a promising method for LMs to efficiently "forget" sensitive training content and mitigate knowledge leakage risks. However, despite its good intentions, could the unlearning mechanism be counterproductive? In this paper, we propose the Textual Unlearning Leakage Attack (TULA), where an adversary can infer information about the unlearned data only by accessing the models before and after unlearning. Furthermore, we present variants of TULA in both black-box and white-box scenarios. Through various experimental results, we critically demonstrate that machine unlearning amplifies the risk of knowledge leakage from LMs. Specifically, TULA can increase an adversary's ability to infer membership information about the unlearned data by more than 20% in black-box scenario. Moreover, TULA can even reconstruct the unlearned data directly with more than 60% accuracy with white-box access. Our work is the first to reveal that machine unlearning in LMs can inversely create greater knowledge risks and inspire the development of more secure unlearning mechanisms.

摘要: 语言模型(LMS)很容易“记忆”训练数据，包括大量私人或受版权保护的内容。为了保护被遗忘的权利，机器遗忘已经成为学习管理系统有效忘记敏感训练内容和降低知识泄漏风险的一种很有前途的方法。然而，尽管这种遗忘机制的用意是好的，但它会适得其反吗？在本文中，我们提出了文本遗忘泄漏攻击(Tula)，在该攻击中，攻击者只能通过访问遗忘前后的模型来推断关于未学习数据的信息。此外，我们还介绍了Tula在黑盒和白盒场景中的变体。通过各种实验结果，我们批判性地证明了机器遗忘放大了最小二乘系统的知识泄漏风险。具体地说，在黑盒情况下，Tula可以将对手推断未学习数据的成员信息的能力提高20%以上。此外，图拉甚至可以通过白盒访问直接重建未学习的数据，准确率超过60%。我们的工作首次揭示了LMS中的机器遗忘可以相反地创造更大的知识风险，并激励更安全的遗忘机制的发展。



## **20. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

MM-SafetyBench：多模式大型语言模型安全评估的基准 cs.CV

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2311.17600v5) [paper-pdf](http://arxiv.org/pdf/2311.17600v5)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at https://github.com/isXinLiu/MM-SafetyBench

摘要: 围绕大语言模型的安全问题已经得到了广泛的研究，但多模式大语言模型的安全性仍未得到充分的研究。在本文中，我们观察到多模式大型语言模型(MLLMS)很容易被与查询相关的图像破坏，就好像文本查询本身是恶意的一样。为了解决这一问题，我们引入了MM-SafetyBch，这是一个全面的框架，旨在针对此类基于图像的操作对MLLMS进行安全关键评估。我们汇编了一个包含13个场景的数据集，总共产生了5,040个文本-图像对。我们对12种最先进型号的分析表明，即使配备的LLM已经安全对准，MLLM也容易受到我们的方法引发的漏洞的影响。对此，我们提出了一种简单而有效的提示策略，以增强MLLMS对这些类型攻击的弹性。我们的工作强调了需要齐心协力加强和改进开放源码MLLM的安全措施，以防范潜在的恶意利用。该资源可在https://github.com/isXinLiu/MM-SafetyBench上获得



## **21. Assessing AI vs Human-Authored Spear Phishing SMS Attacks: An Empirical Study Using the TRAPD Method**

评估人工智能与人类发起的鱼叉网络钓鱼短信攻击：使用TRAPD方法的实证研究 cs.CY

18 pages, 5 figures, 1 table

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.13049v1) [paper-pdf](http://arxiv.org/pdf/2406.13049v1)

**Authors**: Jerson Francia, Derek Hansen, Ben Schooley, Matthew Taylor, Shydra Murray, Greg Snow

**Abstract**: This paper explores the rising concern of utilizing Large Language Models (LLMs) in spear phishing message generation, and their performance compared to human-authored counterparts. Our pilot study compares the effectiveness of smishing (SMS phishing) messages created by GPT-4 and human authors, which have been personalized to willing targets. The targets assessed the messages in a modified ranked-order experiment using a novel methodology we call TRAPD (Threshold Ranking Approach for Personalized Deception). Specifically, targets provide personal information (job title and location, hobby, item purchased online), spear smishing messages are created using this information by humans and GPT-4, targets are invited back to rank-order 12 messages from most to least convincing (and identify which they would click on), and then asked questions about why they ranked messages the way they did. They also guess which messages are created by an LLM and their reasoning. Results from 25 targets show that LLM-generated messages are most often perceived as more convincing than those authored by humans, with messages related to jobs being the most convincing. We characterize different criteria used when assessing the authenticity of messages including word choice, style, and personal relevance. Results also show that targets were unable to identify whether the messages was AI-generated or human-authored and struggled to identify criteria to use in order to make this distinction. This study aims to highlight the urgent need for further research and improved countermeasures against personalized AI-enabled social engineering attacks.

摘要: 本文探讨了在鱼叉式网络钓鱼消息生成中使用大型语言模型(LLM)日益受到关注，以及它们与人类创作的同类消息相比的性能。我们的试点研究比较了GPT-4和人类作者创建的Smish(短信钓鱼)消息的有效性，这些消息已经针对自愿的目标进行了个性化。目标在一种改进的排序实验中使用了一种新的方法来评估消息，我们称之为TRAPD(个性化欺骗的阈值排序方法)。具体地说，目标提供个人信息(职称和地点、爱好、在线购买的物品)，人类和GPT-4使用这些信息创建刺绣信息，邀请目标重新对12条消息按从最有说服力到最不令人信服的顺序进行排序(并确定他们会点击哪些消息)，然后被问及为什么会这样对消息进行排序。他们还猜测哪些消息是由LLM创建的，以及他们的推理。来自25个目标的结果显示，LLM生成的信息通常被认为比人类创作的信息更有说服力，其中与工作有关的信息最令人信服。我们对评估信息真实性时使用的不同标准进行了描述，包括词语选择、风格和个人相关性。结果还显示，目标无法识别这些消息是人工智能生成的还是人类创作的，并且难以确定用于区分的标准。这项研究旨在突出针对个性化人工智能启用的社会工程攻击的进一步研究和改进对策的迫切需要。



## **22. SHIELD: Evaluation and Defense Strategies for Copyright Compliance in LLM Text Generation**

SHIELD：LLM文本生成中版权合规性的评估和防御策略 cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12975v1) [paper-pdf](http://arxiv.org/pdf/2406.12975v1)

**Authors**: Xiaoze Liu, Ting Sun, Tianyang Xu, Feijie Wu, Cunxiang Wang, Xiaoqian Wang, Jing Gao

**Abstract**: Large Language Models (LLMs) have transformed machine learning but raised significant legal concerns due to their potential to produce text that infringes on copyrights, resulting in several high-profile lawsuits. The legal landscape is struggling to keep pace with these rapid advancements, with ongoing debates about whether generated text might plagiarize copyrighted materials. Current LLMs may infringe on copyrights or overly restrict non-copyrighted texts, leading to these challenges: (i) the need for a comprehensive evaluation benchmark to assess copyright compliance from multiple aspects; (ii) evaluating robustness against safeguard bypassing attacks; and (iii) developing effective defenses targeted against the generation of copyrighted text. To tackle these challenges, we introduce a curated dataset to evaluate methods, test attack strategies, and propose lightweight, real-time defenses to prevent the generation of copyrighted text, ensuring the safe and lawful use of LLMs. Our experiments demonstrate that current LLMs frequently output copyrighted text, and that jailbreaking attacks can significantly increase the volume of copyrighted output. Our proposed defense mechanisms significantly reduce the volume of copyrighted text generated by LLMs by effectively refusing malicious requests. Code is publicly available at https://github.com/xz-liu/SHIELD

摘要: 大型语言模型(LLM)改变了机器学习，但由于它们有可能产生侵犯版权的文本，因此引发了重大的法律担忧，导致了几起备受瞩目的诉讼。法律界正在努力跟上这些快速进步的步伐，关于生成的文本是否可能抄袭受版权保护的材料的争论仍在继续。目前的LLM可能侵犯版权或过度限制非版权文本，导致以下挑战：(I)需要一个全面的评估基准，从多个方面评估版权合规性；(Ii)评估对绕过保护措施的攻击的稳健性；以及(Iii)针对版权文本的生成开发有效的防御措施。为了应对这些挑战，我们引入了一个经过精心策划的数据集来评估方法，测试攻击策略，并提出了轻量级、实时的防御措施，以防止版权文本的生成，确保LLMS的安全和合法使用。我们的实验表明，当前的LLM频繁地输出受版权保护的文本，越狱攻击可以显著增加受版权保护的输出量。我们提出的防御机制通过有效地拒绝恶意请求，显著减少了LLMS生成的受版权保护的文本的数量。代码可在https://github.com/xz-liu/SHIELD上公开获得



## **23. Stealth edits for provably fixing or attacking large language models**

用于可证明修复或攻击大型语言模型的隐形编辑 cs.AI

24 pages, 9 figures. Open source implementation:  https://github.com/qinghua-zhou/stealth-edits

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12670v1) [paper-pdf](http://arxiv.org/pdf/2406.12670v1)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Wei Wang, Desmond J. Higham, Alexander N. Gorban, Alexander Bastounis, Ivan Y. Tyukin

**Abstract**: We reveal new methods and the theoretical foundations of techniques for editing large language models. We also show how the new theory can be used to assess the editability of models and to expose their susceptibility to previously unknown malicious attacks. Our theoretical approach shows that a single metric (a specific measure of the intrinsic dimensionality of the model's features) is fundamental to predicting the success of popular editing approaches, and reveals new bridges between disparate families of editing methods. We collectively refer to these approaches as stealth editing methods, because they aim to directly and inexpensively update a model's weights to correct the model's responses to known hallucinating prompts without otherwise affecting the model's behaviour, without requiring retraining. By carefully applying the insight gleaned from our theoretical investigation, we are able to introduce a new network block -- named a jet-pack block -- which is optimised for highly selective model editing, uses only standard network operations, and can be inserted into existing networks. The intrinsic dimensionality metric also determines the vulnerability of a language model to a stealth attack: a small change to a model's weights which changes its response to a single attacker-chosen prompt. Stealth attacks do not require access to or knowledge of the model's training data, therefore representing a potent yet previously unrecognised threat to redistributed foundation models. They are computationally simple enough to be implemented in malware in many cases. Extensive experimental results illustrate and support the method and its theoretical underpinnings. Demos and source code for editing language models are available at https://github.com/qinghua-zhou/stealth-edits.

摘要: 我们揭示了编辑大型语言模型的新方法和技术的理论基础。我们还展示了如何使用新的理论来评估模型的可编辑性，并暴露它们对以前未知的恶意攻击的敏感性。我们的理论方法表明，单一指标(模型特征内在维度的特定衡量标准)是预测流行编辑方法成功的基础，并揭示了不同编辑方法家族之间的新桥梁。我们将这些方法统称为隐形编辑方法，因为它们旨在直接且廉价地更新模型的权重，以纠正模型对已知幻觉提示的反应，而不会以其他方式影响模型的行为，而不需要重新培训。通过仔细应用从我们的理论研究中收集到的见解，我们能够引入一种新的网络块--命名为JET-PACK块--它针对高度选择性的模型编辑进行了优化，仅使用标准的网络操作，并且可以插入到现有网络中。固有的维度度量还决定了语言模型对隐形攻击的脆弱性：对模型权重的微小更改会改变其对攻击者选择的单个提示的响应。隐形攻击不需要访问或了解模型的训练数据，因此对重新分布的基础模型构成了一个以前未被认识到的强大威胁。它们在计算上足够简单，在许多情况下可以在恶意软件中实现。大量的实验结果说明和支持了该方法及其理论基础。有关编辑语言模型的演示和源代码，请访问https://github.com/qinghua-zhou/stealth-edits.



## **24. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

多语言机器生成文本检测中的作者混淆 cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2401.07867v2) [paper-pdf](http://arxiv.org/pdf/2401.07867v2)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).

摘要: 最近的大型语言模型(LLM)的高质量文本生成能力引起了人们对它们的滥用(例如，在大规模生成/传播虚假信息中)的担忧。机器生成文本(MGT)检测对于应对此类威胁非常重要。然而，它容易受到作者身份混淆(AO)方法的影响，例如转译，这可能导致MGTS逃避检测。到目前为止，这只在单一语言环境中进行了评估。因此，最近提出的多语言检测器的敏感性仍然未知。我们通过全面基准测试10种著名的AO方法的性能来填补这一空白，针对11种语言的MGT攻击37种MGT检测方法(即，10$\乘以$37$\乘以$11=4,070个组合)。我们还使用混淆文本来评估数据增强对对手健壮性的影响。结果表明，在所有被测语言中，所有被测试的声学方法都可以逃避自动检测，其中同形文字攻击尤其成功。然而，一些AO方法严重损坏了文本，使其不再可读或不再容易被人类识别(例如，改变语言、奇怪的字符)。



## **25. Can We Trust Large Language Models Generated Code? A Framework for In-Context Learning, Security Patterns, and Code Evaluations Across Diverse LLMs**

我们可以信任大型语言模型生成的代码吗？跨各种LLM的上下文学习、安全模式和代码评估框架 cs.CR

27 pages, Standard Journal Paper submitted to Q1 Elsevier

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12513v1) [paper-pdf](http://arxiv.org/pdf/2406.12513v1)

**Authors**: Ahmad Mohsin, Helge Janicke, Adrian Wood, Iqbal H. Sarker, Leandros Maglaras, Naeem Janjua

**Abstract**: Large Language Models (LLMs) such as ChatGPT and GitHub Copilot have revolutionized automated code generation in software engineering. However, as these models are increasingly utilized for software development, concerns have arisen regarding the security and quality of the generated code. These concerns stem from LLMs being primarily trained on publicly available code repositories and internet-based textual data, which may contain insecure code. This presents a significant risk of perpetuating vulnerabilities in the generated code, creating potential attack vectors for exploitation by malicious actors. Our research aims to tackle these issues by introducing a framework for secure behavioral learning of LLMs through In-Content Learning (ICL) patterns during the code generation process, followed by rigorous security evaluations. To achieve this, we have selected four diverse LLMs for experimentation. We have evaluated these coding LLMs across three programming languages and identified security vulnerabilities and code smells. The code is generated through ICL with curated problem sets and undergoes rigorous security testing to evaluate the overall quality and trustworthiness of the generated code. Our research indicates that ICL-driven one-shot and few-shot learning patterns can enhance code security, reducing vulnerabilities in various programming scenarios. Developers and researchers should know that LLMs have a limited understanding of security principles. This may lead to security breaches when the generated code is deployed in production systems. Our research highlights LLMs are a potential source of new vulnerabilities to the software supply chain. It is important to consider this when using LLMs for code generation. This research article offers insights into improving LLM security and encourages proactive use of LLMs for code generation to ensure software system safety.

摘要: ChatGPT和GitHub Copilot等大型语言模型(LLM)彻底改变了软件工程中的自动代码生成。然而，随着这些模型越来越多地用于软件开发，产生了对所生成代码的安全性和质量的担忧。这些担忧源于LLM主要接受关于公开可用的代码库和基于互联网的文本数据的培训，这些数据可能包含不安全的代码。这带来了使生成的代码中的漏洞永久化的重大风险，从而创建了潜在的攻击载体，供恶意攻击者利用。我们的研究旨在通过引入一个框架来解决这些问题，该框架在代码生成过程中通过内容内学习(ICL)模式来实现LLM的安全行为学习，然后进行严格的安全评估。为了实现这一点，我们选择了四种不同的LLM进行实验。我们已经在三种编程语言中评估了这些编码LLM，并确定了安全漏洞和代码气味。代码是通过ICL生成的，带有精选的问题集，并经过严格的安全测试，以评估生成的代码的整体质量和可信度。我们的研究表明，ICL驱动的一次和几次学习模式可以增强代码安全性，减少各种编程场景中的漏洞。开发人员和研究人员应该知道，LLM对安全原则的理解有限。当生成的代码部署在生产系统中时，这可能会导致安全漏洞。我们的研究强调，LLM是软件供应链新漏洞的潜在来源。在使用LLM进行代码生成时，考虑这一点非常重要。这篇研究文章提供了改进LLM安全性的见解，并鼓励主动使用LLM进行代码生成，以确保软件系统安全。



## **26. Identifying and Mitigating Privacy Risks Stemming from Language Models: A Survey**

识别和缓解源于语言模型的隐私风险：一项调查 cs.CL

15 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2310.01424v2) [paper-pdf](http://arxiv.org/pdf/2310.01424v2)

**Authors**: Victoria Smith, Ali Shahin Shamsabadi, Carolyn Ashurst, Adrian Weller

**Abstract**: Large Language Models (LLMs) have shown greatly enhanced performance in recent years, attributed to increased size and extensive training data. This advancement has led to widespread interest and adoption across industries and the public. However, training data memorization in Machine Learning models scales with model size, particularly concerning for LLMs. Memorized text sequences have the potential to be directly leaked from LLMs, posing a serious threat to data privacy. Various techniques have been developed to attack LLMs and extract their training data. As these models continue to grow, this issue becomes increasingly critical. To help researchers and policymakers understand the state of knowledge around privacy attacks and mitigations, including where more work is needed, we present the first SoK on data privacy for LLMs. We (i) identify a taxonomy of salient dimensions where attacks differ on LLMs, (ii) systematize existing attacks, using our taxonomy of dimensions to highlight key trends, (iii) survey existing mitigation strategies, highlighting their strengths and limitations, and (iv) identify key gaps, demonstrating open problems and areas for concern.

摘要: 近年来，由于规模的增加和大量的训练数据，大型语言模型(LLM)的性能得到了极大的提高。这一进步引起了业界和公众的广泛兴趣和采用。然而，机器学习模型中的训练数据记忆随模型的大小而变化，尤其是对于LLMS。记忆的文本序列有可能直接从LLMS泄露，对数据隐私构成严重威胁。已经开发了各种技术来攻击LLMS并提取它们的训练数据。随着这些模式的不断发展，这个问题变得越来越关键。为了帮助研究人员和政策制定者了解有关隐私攻击和缓解的知识状况，包括需要更多工作的地方，我们提出了第一个关于低成本管理的数据隐私的SOK。我们(I)确定针对LLMS的攻击不同的显著维度的分类，(Ii)系统化现有攻击，使用我们的维度分类来突出关键趋势，(Iii)调查现有缓解策略，突出其优势和局限性，以及(Iv)确定关键差距，展示公开的问题和值得关注的领域。



## **27. Unique Security and Privacy Threats of Large Language Model: A Comprehensive Survey**

大型语言模型的独特安全和隐私威胁：全面调查 cs.CR

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.07973v2) [paper-pdf](http://arxiv.org/pdf/2406.07973v2)

**Authors**: Shang Wang, Tianqing Zhu, Bo Liu, Ming Ding, Xu Guo, Dayong Ye, Wanlei Zhou, Philip S. Yu

**Abstract**: With the rapid development of artificial intelligence, large language models (LLMs) have made remarkable advancements in natural language processing. These models are trained on vast datasets to exhibit powerful language understanding and generation capabilities across various applications, including machine translation, chatbots, and agents. However, LLMs have revealed a variety of privacy and security issues throughout their life cycle, drawing significant academic and industrial attention. Moreover, the risks faced by LLMs differ significantly from those encountered by traditional language models. Given that current surveys lack a clear taxonomy of unique threat models across diverse scenarios, we emphasize the unique privacy and security threats associated with five specific scenarios: pre-training, fine-tuning, retrieval-augmented generation systems, deployment, and LLM-based agents. Addressing the characteristics of each risk, this survey outlines potential threats and countermeasures. Research on attack and defense situations can offer feasible research directions, enabling more areas to benefit from LLMs.

摘要: 随着人工智能的快速发展，大语言模型在自然语言处理方面取得了显著的进步。这些模型是在海量数据集上进行训练的，以展示强大的语言理解和跨各种应用程序的生成能力，包括机器翻译、聊天机器人和代理。然而，LLMS在其整个生命周期中暴露了各种隐私和安全问题，引起了学术界和工业界的极大关注。此外，LLMS面临的风险与传统语言模型所遇到的风险有很大不同。鉴于目前的调查缺乏针对不同场景的独特威胁模型的明确分类，我们强调了与五种特定场景相关的独特隐私和安全威胁：预培训、微调、检索增强生成系统、部署和基于LLM的代理。针对每个风险的特点，本调查概述了潜在的威胁和对策。对攻防态势的研究可以提供可行的研究方向，使更多的地区受益于低成本管理。



## **28. Defending Against Social Engineering Attacks in the Age of LLMs**

在法学硕士时代防御社会工程攻击 cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12263v1) [paper-pdf](http://arxiv.org/pdf/2406.12263v1)

**Authors**: Lin Ai, Tharindu Kumarage, Amrita Bhattacharjee, Zizhou Liu, Zheng Hui, Michael Davinroy, James Cook, Laura Cassani, Kirill Trapeznikov, Matthias Kirchner, Arslan Basharat, Anthony Hoogs, Joshua Garland, Huan Liu, Julia Hirschberg

**Abstract**: The proliferation of Large Language Models (LLMs) poses challenges in detecting and mitigating digital deception, as these models can emulate human conversational patterns and facilitate chat-based social engineering (CSE) attacks. This study investigates the dual capabilities of LLMs as both facilitators and defenders against CSE threats. We develop a novel dataset, SEConvo, simulating CSE scenarios in academic and recruitment contexts, and designed to examine how LLMs can be exploited in these situations. Our findings reveal that, while off-the-shelf LLMs generate high-quality CSE content, their detection capabilities are suboptimal, leading to increased operational costs for defense. In response, we propose ConvoSentinel, a modular defense pipeline that improves detection at both the message and the conversation levels, offering enhanced adaptability and cost-effectiveness. The retrieval-augmented module in ConvoSentinel identifies malicious intent by comparing messages to a database of similar conversations, enhancing CSE detection at all stages. Our study highlights the need for advanced strategies to leverage LLMs in cybersecurity.

摘要: 大型语言模型(LLM)的激增给检测和减轻数字欺骗带来了挑战，因为这些模型可以模拟人类的对话模式，并促进基于聊天的社会工程(CSE)攻击。本研究探讨低层管理人员作为CSE威胁的促进者和防御者的双重能力。我们开发了一个新的数据集SEConvo，模拟了学术和招聘环境中的CSE场景，并旨在研究如何在这些情况下利用LLM。我们的发现表明，虽然现成的LLM可以生成高质量的CSE内容，但它们的检测能力并不理想，从而导致防御操作成本增加。作为回应，我们提出了ConvoSentinel，这是一种模块化的防御管道，可以同时改进消息和会话级别的检测，提供更强的适应性和成本效益。ConvoSentinel中的检索增强模块通过将消息与类似对话的数据库进行比较来识别恶意意图，从而增强了所有阶段的CSE检测。我们的研究强调了在网络安全中利用低成本管理的高级战略的必要性。



## **29. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12259v1) [paper-pdf](http://arxiv.org/pdf/2406.12259v1)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。我们发现，虽然整合对抗性数据并不会显著降低医学基准上的整体模型性能，但它确实会导致微调模型权重的显著变化，这表明了一条检测和对抗模型攻击的潜在路径。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **30. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models**

CleanGen：缓解大型语言模型中生成任务的后门攻击 cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12257v1) [paper-pdf](http://arxiv.org/pdf/2406.12257v1)

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CleanGen, to mitigate backdoor attacks for generation tasks in LLMs. CleanGenis a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CleanGen is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CleanGen to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CleanGen against five SOTA backdoor attacks. Our results show that CleanGen achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CleanGen maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.

摘要: 大型语言模型(LLM)在生成任务中的出色性能使实践者能够利用公开可用的模型来支持定制应用程序，如聊天机器人和虚拟助手。然而，用于训练或微调这些LLM的数据往往是秘密的，这使得攻击者能够危害数据并向模型注入后门。本文提出了一种新的推理时间防御机制CleanGen，用于缓解LLMS中针对生成任务的后门攻击。CleanGenis是一种轻量级且有效的解码策略，与最先进的(SOTA)LLM兼容。我们在CleanGen背后的见解是，与其他LLM相比，反向LLM向代表攻击者所需内容的令牌分配的概率要高得多。令牌概率中的这些差异使CleanGen能够识别攻击者偏爱的可疑令牌，并将其替换为由另一个未被同一攻击者破解的LLM生成的令牌，从而避免生成攻击者所需的内容。我们对CleanGen进行了五次Sota后门攻击评估。我们的结果显示，对于所有五个后门攻击，CleanGen实现的攻击成功率(ASR)都低于五个SOTA基线防御。此外，部署CleanGen的LLMS在以最小的额外计算开销服务于良性用户查询时，在其响应中保持了帮助。



## **31. Privacy-Preserved Neural Graph Databases**

隐私保护的神经图数据库 cs.DB

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.15591v5) [paper-pdf](http://arxiv.org/pdf/2312.15591v5)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, Yangqiu Song

**Abstract**: In the era of large language models (LLMs), efficient and accurate data retrieval has become increasingly crucial for the use of domain-specific or private data in the retrieval augmented generation (RAG). Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (GDBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data which can be adaptively trained with LLMs. The usage of neural embedding storage and Complex neural logical Query Answering (CQA) provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the domain-specific or private databases. Malicious attackers can infer more sensitive information in the database using well-designed queries such as from the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training stage due to the privacy concerns. In this work, we propose a privacy-preserved neural graph database (P-NGDB) framework to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to enforce the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries.

摘要: 在大型语言模型(LLMS)时代，高效和准确的数据检索对于在检索增强生成(RAG)中使用特定领域或私有数据变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(GDB)和神经网络的优点，能够有效地存储、检索和分析图结构的数据，这些数据可以用LLMS进行自适应训练。神经嵌入存储和复杂神经逻辑查询应答(CQA)的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。然而，这种能力是有内在权衡的，因为它会给特定于域或私有的数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的查询来推断数据库中更敏感的信息，例如从图灵奖获得者1950年前和1940年后出生的地方的答案集中，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，居住地可能在培训阶段已被删除。在这项工作中，我们提出了一个隐私保护的神经图库(P-NGDB)框架，以缓解NGDB中隐私泄露的风险。在训练阶段引入对抗性训练技术，强制NGDB在查询私有信息时产生不可区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。



## **32. JailGuard: A Universal Detection Framework for LLM Prompt-based Attacks**

JailGuard：针对LLM基于预算的攻击的通用检测框架 cs.CR

28 pages, 9 figures

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.10766v3) [paper-pdf](http://arxiv.org/pdf/2312.10766v3)

**Authors**: Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Ming Hu, Jie Zhang, Yang Liu, Shiqing Ma, Chao Shen

**Abstract**: Large Language Models (LLMs) and Multi-Modal LLMs (MLLMs) have played a critical role in numerous applications. However, current LLMs are vulnerable to prompt-based attacks, with jailbreaking attacks enabling LLMs to generate harmful content, while hijacking attacks manipulate the model to perform unintended tasks, underscoring the necessity for detection methods. Unfortunately, existing detecting approaches are usually tailored to specific attacks, resulting in poor generalization in detecting various attacks across different modalities. To address it, we propose JailGuard, a universal detection framework for jailbreaking and hijacking attacks across LLMs and MLLMs. JailGuard operates on the principle that attacks are inherently less robust than benign ones, regardless of method or modality. Specifically, JailGuard mutates untrusted inputs to generate variants and leverages the discrepancy of the variants' responses on the model to distinguish attack samples from benign samples. We implement 18 mutators for text and image inputs and design a mutator combination policy to further improve detection generalization. To evaluate the effectiveness of JailGuard, we build the first comprehensive multi-modal attack dataset, containing 11,000 data items across 15 known attack types. The evaluation suggests that JailGuard achieves the best detection accuracy of 86.14%/82.90% on text and image inputs, outperforming state-of-the-art methods by 11.81%-25.73% and 12.20%-21.40%.

摘要: 大语言模型(LLM)和多模式LLM(MLLM)在许多应用中发挥了关键作用。然而，当前的LLM容易受到基于提示的攻击，越狱攻击使LLM能够生成有害内容，而劫持攻击操纵模型执行非预期任务，这突显了检测方法的必要性。遗憾的是，现有的检测方法通常是针对特定的攻击量身定做的，导致在检测不同模式的各种攻击时通用性较差。为了解决这个问题，我们提出了JailGuard，这是一个通用的检测框架，用于跨LLMS和MLLMS的越狱和劫持攻击。JailGuard的运作原则是，无论方法或方式如何，攻击天生就不如良性攻击那么强大。具体地说，JailGuard会变异不可信的输入以生成变体，并利用变体对模型的响应差异来区分攻击样本和良性样本。我们为文本和图像输入实现了18个变异器，并设计了变异器组合策略，进一步提高了检测的泛化能力。为了评估JailGuard的有效性，我们构建了第一个全面的多模式攻击数据集，包含15种已知攻击类型的11,000个数据项。评估表明，JailGuard对文本和图像输入的检测准确率达到了86.14%/82.90%，分别比最先进的方法高出11.81%-25.73%和12.20%-21.40%。



## **33. Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**

（几乎）免费进行安全微调：Vision大型语言模型的基线 cs.LG

ICML 2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2402.02207v2) [paper-pdf](http://arxiv.org/pdf/2402.02207v2)

**Authors**: Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales

**Abstract**: Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset are available at https://github.com/ys-zong/VLGuard.

摘要: 目前的VISION大型语言模型(VLLM)显示出非凡的能力，但很容易产生有害内容，甚至容易受到最简单的越狱攻击。我们的初步分析发现，这是由于视觉语言教学微调过程中存在有害数据，而VLLM微调可能会导致忘记支持LLM之前学习的安全对齐。为了解决这个问题，我们首先策划了一个视觉-语言安全的指令遵循数据集VLGuard，涵盖了各种有害类别。我们的实验表明，将该数据集集成到标准视觉语言微调中或将其用于后自组织微调，可以有效地安全地对齐VLLM。这种对齐是在对模型的帮助最小的影响甚至是增强的情况下实现的。我们的安全微调数据集的多功能性使其成为安全测试现有VLLM、培训新模型或保护预先培训的VLLM的宝贵资源。实验结果表明，微调的VLLM有效地拒绝了不安全的指令，并显著降低了几种黑盒对抗攻击的成功率，这些攻击在许多情况下接近于零。代码和数据集可在https://github.com/ys-zong/VLGuard.上获得



## **34. MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance**

MLLM-保护者：确保MLLM的安全而不损害绩效 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2401.02906v3) [paper-pdf](http://arxiv.org/pdf/2401.02906v3)

**Authors**: Renjie Pi, Tianyang Han, Jianshu Zhang, Yueqi Xie, Rui Pan, Qing Lian, Hanze Dong, Jipeng Zhang, Tong Zhang

**Abstract**: The deployment of multimodal large language models (MLLMs) has brought forth a unique vulnerability: susceptibility to malicious attacks through visual inputs. This paper investigates the novel challenge of defending MLLMs against such attacks. Compared to large language models (LLMs), MLLMs include an additional image modality. We discover that images act as a ``foreign language" that is not considered during safety alignment, making MLLMs more prone to producing harmful responses. Unfortunately, unlike the discrete tokens considered in text-based LLMs, the continuous nature of image signals presents significant alignment challenges, which poses difficulty to thoroughly cover all possible scenarios. This vulnerability is exacerbated by the fact that most state-of-the-art MLLMs are fine-tuned on limited image-text pairs that are much fewer than the extensive text-based pretraining corpus, which makes the MLLMs more prone to catastrophic forgetting of their original abilities during safety fine-tuning. To tackle these challenges, we introduce MLLM-Protector, a plug-and-play strategy that solves two subtasks: 1) identifying harmful responses via a lightweight harm detector, and 2) transforming harmful responses into harmless ones via a detoxifier. This approach effectively mitigates the risks posed by malicious visual inputs without compromising the original performance of MLLMs. Our results demonstrate that MLLM-Protector offers a robust solution to a previously unaddressed aspect of MLLM security.

摘要: 多模式大型语言模型(MLLMS)的部署带来了一个独特的漏洞：通过视觉输入易受恶意攻击。本文研究了防御MLLMS免受此类攻击的新挑战。与大型语言模型(LLM)相比，MLLM包括一种额外的图像通道。我们发现，图像作为一种“外语”在安全对准过程中没有被考虑，使得MLLMS更容易产生有害的反应。不幸的是，与基于文本的LLMS中考虑的离散标记不同，图像信号的连续性质带来了巨大的对齐挑战，这使得很难完全覆盖所有可能的场景。大多数最先进的MLLS都是在有限的图文对上进行微调的，这比基于大量文本的预训练语料库要少得多，这使得MLLMS在安全微调期间更容易灾难性地忘记其原始能力，这加剧了这一漏洞。为了应对这些挑战，我们引入了MLLM-Protector，这是一种即插即用策略，可以解决两个子任务：1)通过轻型伤害检测器识别有害反应，2)通过解毒器将有害反应转化为无害反应。这种方法有效地降低了恶意视觉输入带来的风险，而不会影响MLLMS的原始性能。我们的结果表明，MLLM-Protector为MLLM安全的一个以前未解决的方面提供了一个健壮的解决方案。



## **35. Knowledge-to-Jailbreak: One Knowledge Point Worth One Attack**

知识越狱：一个知识点值得一次攻击 cs.CL

18 pages, 14 figures, 11 tables

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11682v1) [paper-pdf](http://arxiv.org/pdf/2406.11682v1)

**Authors**: Shangqing Tu, Zhuoran Pan, Wenxuan Wang, Zhexin Zhang, Yuliang Sun, Jifan Yu, Hongning Wang, Lei Hou, Juanzi Li

**Abstract**: Large language models (LLMs) have been increasingly applied to various domains, which triggers increasing concerns about LLMs' safety on specialized domains, e.g. medicine. However, testing the domain-specific safety of LLMs is challenging due to the lack of domain knowledge-driven attacks in existing benchmarks. To bridge this gap, we propose a new task, knowledge-to-jailbreak, which aims to generate jailbreaks from domain knowledge to evaluate the safety of LLMs when applied to those domains. We collect a large-scale dataset with 12,974 knowledge-jailbreak pairs and fine-tune a large language model as jailbreak-generator, to produce domain knowledge-specific jailbreaks. Experiments on 13 domains and 8 target LLMs demonstrate the effectiveness of jailbreak-generator in generating jailbreaks that are both relevant to the given knowledge and harmful to the target LLMs. We also apply our method to an out-of-domain knowledge base, showing that jailbreak-generator can generate jailbreaks that are comparable in harmfulness to those crafted by human experts. Data and code: https://github.com/THU-KEG/Knowledge-to-Jailbreak/.

摘要: 大语言模型被越来越多地应用到各个领域，这引发了人们对大语言模型在医学等专业领域的安全性的日益关注。然而，由于现有基准测试中缺乏领域知识驱动的攻击，因此测试LLMS的领域特定安全是具有挑战性的。为了弥补这一差距，我们提出了一个新的任务，知识越狱，其目的是从领域知识生成越狱来评估LLMS应用于这些领域时的安全性。我们收集了一个包含12,974个知识越狱对的大规模数据集，并微调了一个大型语言模型作为越狱生成器，以产生特定于领域知识的越狱。在13个领域和8个目标LLMS上的实验表明，越狱生成器能够有效地生成与给定知识相关且对目标LLMS有害的越狱。我们还将我们的方法应用于域外知识库，表明越狱生成器可以生成与人类专家创建的越狱在危害性上相当的越狱。数据和代码：https://github.com/THU-KEG/Knowledge-to-Jailbreak/.



## **36. Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**

Bileve：通过双层签名保护大型语言模型中的文本出处，防止欺骗 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.01946v2) [paper-pdf](http://arxiv.org/pdf/2406.01946v2)

**Authors**: Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren

**Abstract**: Text watermarks for large language models (LLMs) have been commonly used to identify the origins of machine-generated content, which is promising for assessing liability when combating deepfake or harmful content. While existing watermarking techniques typically prioritize robustness against removal attacks, unfortunately, they are vulnerable to spoofing attacks: malicious actors can subtly alter the meanings of LLM-generated responses or even forge harmful content, potentially misattributing blame to the LLM developer. To overcome this, we introduce a bi-level signature scheme, Bileve, which embeds fine-grained signature bits for integrity checks (mitigating spoofing attacks) as well as a coarse-grained signal to trace text sources when the signature is invalid (enhancing detectability) via a novel rank-based sampling strategy. Compared to conventional watermark detectors that only output binary results, Bileve can differentiate 5 scenarios during detection, reliably tracing text provenance and regulating LLMs. The experiments conducted on OPT-1.3B and LLaMA-7B demonstrate the effectiveness of Bileve in defeating spoofing attacks with enhanced detectability.

摘要: 大型语言模型(LLM)的文本水印通常用于识别机器生成内容的来源，这有望在打击深度虚假或有害内容时评估责任。虽然现有的水印技术通常将健壮性放在免受删除攻击的优先位置，但不幸的是，它们容易受到欺骗性攻击：恶意行为者可以巧妙地更改LLM生成的响应的含义，甚至伪造有害内容，可能会将责任错误地归咎于LLM开发人员。为了克服这一问题，我们提出了一种双层签名方案BiLEVE，该方案通过一种新颖的基于等级的采样策略嵌入细粒度的签名比特用于完整性检查(缓解欺骗攻击)，并在签名无效时嵌入粗粒度的信号来跟踪文本来源(增强了可检测性)。与传统的只输出二进制结果的水印检测器相比，BiLEVE在检测过程中可以区分5种场景，可靠地追踪文本来源和规范LLM。在OPT-1.3B和LLAMA-7B上进行的实验证明了BiLEVE在抵抗欺骗攻击方面的有效性，并增强了可检测性。



## **37. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

ART：文本到图像模型的自动红色团队以保护良性用户 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2405.19360v2) [paper-pdf](http://arxiv.org/pdf/2405.19360v2)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.

摘要: 由于具有创造内容的能力，大规模的预先训练的生成性模型正在席卷世界。同时，为了保护用户的权利和安全，制定了对这些生成模型的保障措施，其中大部分是为大型语言模型设计的。现有的方法主要针对越狱和对抗性攻击，主要是在恶意提示下对模型的安全性进行评估。最近的研究发现，手动创建的安全提示可能会无意中引发不安全的世代。为了进一步系统地评估文本到图像模型的安全风险，我们提出了一个新的自动红色团队框架ART。我们的方法利用视觉语言模型和大型语言模型来建立不安全生成及其提示之间的联系，从而更有效地识别模型的漏洞。通过我们的综合实验，我们揭示了流行的开源文本到图像模型的毒性。实验也验证了ART的有效性、适应性和多样性。此外，我们还介绍了三个大型红团队数据集，用于研究与文本到图像模型相关的安全风险。数据集和模型可在https://github.com/GuanlinLee/ART.中找到



## **38. $\texttt{MoE-RBench}$: Towards Building Reliable Language Models with Sparse Mixture-of-Experts**

$\textttt {MoE-RBench}$：利用稀疏专家混合构建可靠的语言模型 cs.LG

9 pages, 8 figures, camera ready on ICML2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11353v1) [paper-pdf](http://arxiv.org/pdf/2406.11353v1)

**Authors**: Guanjie Chen, Xinyu Zhao, Tianlong Chen, Yu Cheng

**Abstract**: Mixture-of-Experts (MoE) has gained increasing popularity as a promising framework for scaling up large language models (LLMs). However, the reliability assessment of MoE lags behind its surging applications. Moreover, when transferred to new domains such as in fine-tuning MoE models sometimes underperform their dense counterparts. Motivated by the research gap and counter-intuitive phenomenon, we propose $\texttt{MoE-RBench}$, the first comprehensive assessment of SMoE reliability from three aspects: $\textit{(i)}$ safety and hallucination, $\textit{(ii)}$ resilience to adversarial attacks, and $\textit{(iii)}$ out-of-distribution robustness. Extensive models and datasets are tested to compare the MoE to dense networks from these reliability dimensions. Our empirical observations suggest that with appropriate hyperparameters, training recipes, and inference techniques, we can build the MoE model more reliably than the dense LLM. In particular, we find that the robustness of SMoE is sensitive to the basic training settings. We hope that this study can provide deeper insights into how to adapt the pre-trained MoE model to other tasks with higher-generation security, quality, and stability. Codes are available at https://github.com/UNITES-Lab/MoE-RBench

摘要: 专家混合(MOE)作为一种有前途的扩展大型语言模型(LLM)的框架已经越来越受欢迎。然而，MOE的可靠性评估落后于其激增的应用。此外，当转移到新的领域时，例如在微调的MOE模型中，有时表现不如密集的对应模型。受研究空白和反直觉现象的启发，我们首次从三个方面对SMOE的可靠性进行了全面的评估：安全和幻觉，对对手攻击的恢复能力，以及分布外的稳健性。测试了大量的模型和数据集，以从这些可靠性维度将MoE与密集网络进行比较。我们的经验观察表明，通过适当的超参数、训练配方和推理技术，我们可以建立比密集的LLM更可靠的MOE模型。特别是，我们发现SMOE的稳健性对基本训练设置很敏感。我们希望这项研究能够为如何将预先训练的MOE模型适应于具有更高一代安全性、质量和稳定性的其他任务提供更深层次的见解。有关代码，请访问https://github.com/UNITES-Lab/MoE-RBench



## **39. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

8 pages

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11260v1) [paper-pdf](http://arxiv.org/pdf/2406.11260v1)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.

摘要: 假新闻的传播对个人产生负面影响，被视为需要解决的重大社会挑战。已经确定了许多算法和有洞察力的功能来检测假新闻。然而，随着最近的LLM及其先进一代能力，许多可检测的特征（例如，风格转换攻击）可以被更改，使其与真实新闻区分起来更具挑战性。这项研究提出了对抗性风格增强AdStyle来训练一个假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。我们模型的关键机制是仔细使用LLM来自动生成多样化但连贯的风格转换攻击提示。这改善了检测器特别难以处理的提示的生成。实验表明，当在假新闻基准数据集上进行测试时，我们的增强策略提高了鲁棒性和检测性能。



## **40. Evading AI-Generated Content Detectors using Homoglyphs**

使用同字形躲避人工智能生成的内容检测器 cs.CL

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11239v1) [paper-pdf](http://arxiv.org/pdf/2406.11239v1)

**Authors**: Aldan Creo, Shushanta Pudasaini

**Abstract**: The generation of text that is increasingly human-like has been enabled by the advent of large language models (LLMs). As the detection of AI-generated content holds significant importance in the fight against issues such as misinformation and academic cheating, numerous studies have been conducted to develop reliable LLM detectors. While promising results have been demonstrated by such detectors on test data, recent research has revealed that they can be circumvented by employing different techniques. In this article, homoglyph-based ($a \rightarrow {\alpha}$) attacks that can be used to circumvent existing LLM detectors are presented. The efficacy of the attacks is illustrated by analizing how homoglyphs shift the tokenization of the text, and thus its token loglikelihoods. A comprehensive evaluation is conducted to assess the effectiveness of homoglyphs on state-of-the-art LLM detectors, including Binoculars, DetectGPT, OpenAI's detector, and watermarking techniques, on five different datasets. A significant reduction in the efficiency of all the studied configurations of detectors and datasets, down to an accuracy of 0.5 (random guessing), is demonstrated by the proposed approach. The results show that homoglyph-based attacks can effectively evade existing LLM detectors, and the implications of these findings are discussed along with possible defenses against such attacks.

摘要: 大型语言模型(LLM)的出现使得生成越来越像人类的文本成为可能。由于检测人工智能生成的内容在打击错误信息和学术作弊等问题方面具有重要意义，人们进行了大量研究，以开发可靠的LLM检测器。虽然这种探测器已经在测试数据上证明了有希望的结果，但最近的研究表明，可以通过使用不同的技术来规避这些结果。在这篇文章中，提出了可用于绕过现有LLM检测器的基于同形符号的($a\right tarrow{\alpha}$)攻击。通过分析同形文字如何改变文本的标记化，从而改变其标记性日志可能性，说明了攻击的有效性。在五个不同的数据集上，进行了一项全面的评估，以评估同种文字在最先进的LLM检测器上的有效性，包括双筒望远镜、DetectGPT、OpenAI的检测器和水印技术。通过提出的方法，所有研究的探测器和数据集的配置的效率都显著降低，精度降至0.5(随机猜测)。结果表明，基于同形文字的攻击可以有效地避开现有的LLM检测器，并讨论了这些发现的含义以及对此类攻击的可能防御。



## **41. ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates**

ChatBug：聊天模板引发的对齐LLM的常见漏洞 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.12935v1) [paper-pdf](http://arxiv.org/pdf/2406.12935v1)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, Radha Poovendran

**Abstract**: Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale.   In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research

摘要: 大型语言模型(LLM)应该遵循用户的指示并参与对话。增强LLMS的指令遵循能力的技术通常使用根据预定义的聊天模板构造的数据对其进行微调。尽管聊天模板被证明在优化LLM性能方面是有效的，但人们对它们对LLM安全调整的影响知之甚少，这对于安全地大规模部署LLMS至关重要。在本文中，我们研究了聊天模板如何影响LLMS的安全对齐。我们发现了一个由聊天模板引入的名为ChatBug的常见漏洞。我们识别ChatBug的关键洞察力是，聊天模板提供了一种严格的格式，需要LLMS遵循，而不是用户。因此，恶意用户在提示LLMS时可能不一定遵循聊天模板。相反，恶意用户可以利用他们对聊天模板的了解，并相应地精心编制他们的提示，以绕过LLMS的安全对齐。我们开发了两个攻击来利用ChatBug漏洞。我们演示了恶意用户可以利用8个最先进的(SOTA)LLM的ChatBug漏洞，并有效地从这些模型中引发意外响应。此外，我们发现ChatBug可以被现有的越狱攻击所利用，以提高他们的攻击成功率。我们调查了针对ChatBug的潜在对策。我们的结果表明，虽然对抗性训练有效地缓解了ChatBug漏洞，但受害者模型导致了显著的性能下降。这些结果突显了安全性调整和帮助之间的权衡。开发新的教学调整方法来平衡这种权衡是未来研究的一个开放和关键的方向



## **42. GoldCoin: Grounding Large Language Models in Privacy Laws via Contextual Integrity Theory**

金币：通过上下文完整性理论将大型语言模型作为隐私法的基础 cs.CL

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11149v1) [paper-pdf](http://arxiv.org/pdf/2406.11149v1)

**Authors**: Wei Fan, Haoran Li, Zheye Deng, Weiqi Wang, Yangqiu Song

**Abstract**: Privacy issues arise prominently during the inappropriate transmission of information between entities. Existing research primarily studies privacy by exploring various privacy attacks, defenses, and evaluations within narrowly predefined patterns, while neglecting that privacy is not an isolated, context-free concept limited to traditionally sensitive data (e.g., social security numbers), but intertwined with intricate social contexts that complicate the identification and analysis of potential privacy violations. The advent of Large Language Models (LLMs) offers unprecedented opportunities for incorporating the nuanced scenarios outlined in privacy laws to tackle these complex privacy issues. However, the scarcity of open-source relevant case studies restricts the efficiency of LLMs in aligning with specific legal statutes. To address this challenge, we introduce a novel framework, GoldCoin, designed to efficiently ground LLMs in privacy laws for judicial assessing privacy violations. Our framework leverages the theory of contextual integrity as a bridge, creating numerous synthetic scenarios grounded in relevant privacy statutes (e.g., HIPAA), to assist LLMs in comprehending the complex contexts for identifying privacy risks in the real world. Extensive experimental results demonstrate that GoldCoin markedly enhances LLMs' capabilities in recognizing privacy risks across real court cases, surpassing the baselines on different judicial tasks.

摘要: 隐私问题突出地出现在实体之间不适当的信息传输过程中。现有的研究主要是通过在狭隘的预定义模式中探索各种隐私攻击、防御和评估来研究隐私，而忽略了隐私不是一个孤立的、与上下文无关的概念，仅限于传统的敏感数据(例如，社会安全号码)，而是与错综复杂的社会背景交织在一起，这使得识别和分析潜在的隐私侵犯变得复杂。大型语言模型(LLM)的出现为纳入隐私法中概述的细微差别场景提供了前所未有的机会，以解决这些复杂的隐私问题。然而，开源相关案例研究的匮乏限制了LLMS与具体法律法规保持一致的效率。为了应对这一挑战，我们引入了一个新的框架，GoldCoin，旨在有效地将LLM置于隐私法中，用于司法评估隐私侵权行为。我们的框架利用上下文完整性理论作为桥梁，创建基于相关隐私法规(例如HIPAA)的大量合成场景，以帮助LLMS理解复杂的上下文以识别现实世界中的隐私风险。广泛的实验结果表明，GoldCoin显著增强了LLMS在真实法庭案件中识别隐私风险的能力，超过了不同司法任务的基线。



## **43. Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics**

强调在机器人技术中部署LLM/VLM的安全问题 cs.RO

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2402.10340v4) [paper-pdf](http://arxiv.org/pdf/2402.10340v4)

**Authors**: Xiyang Wu, Souradip Chakraborty, Ruiqi Xian, Jing Liang, Tianrui Guan, Fuxiao Liu, Brian M. Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works focus on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation and navigation. Despite these improvements, analyzing the safety of such systems remains underexplored yet extremely critical. LLMs and VLMs are highly susceptible to adversarial inputs, prompting a significant inquiry into the safety of robotic systems. This concern is important because robotics operate in the physical world where erroneous actions can result in severe consequences. This paper explores this issue thoroughly, presenting a mathematical formulation of potential attacks on LLM/VLM-based robotic systems and offering experimental evidence of the safety challenges. Our empirical findings highlight a significant vulnerability: simple modifications to the input can drastically reduce system effectiveness. Specifically, our results demonstrate an average performance deterioration of 19.4% under minor input prompt modifications and a more alarming 29.1% under slight perceptual changes. These findings underscore the urgent need for robust countermeasures to ensure the safe and reliable deployment of advanced LLM/VLM-based robotic systems.

摘要: 在这篇文章中，我们强调了与将大语言模型(LLM)和视觉语言模型(VLM)集成到机器人应用中相关的健壮性和安全性的关键问题。最近的工作集中在使用LLMS和VLMS来提高机器人任务的性能，如操纵和导航。尽管有了这些改进，分析这类系统的安全性仍然没有得到充分的探索，但仍然非常关键。LLM和VLM非常容易受到敌意输入的影响，这促使人们对机器人系统的安全性进行了重大调查。这一担忧很重要，因为机器人是在物理世界中运行的，在那里错误的行动可能会导致严重的后果。本文对这一问题进行了深入的探讨，给出了对基于LLM/VLM的机器人系统的潜在攻击的数学公式，并提供了安全挑战的实验证据。我们的经验发现突显了一个重大的脆弱性：对输入的简单修改可能会极大地降低系统效率。具体地说，我们的结果显示，在微小的输入提示修改下，性能平均下降了19.4%，而在轻微的感知变化下，性能下降了29.1%。这些发现突显了迫切需要强有力的对策，以确保安全可靠地部署先进的基于LLM/VLM的机器人系统。



## **44. garak: A Framework for Security Probing Large Language Models**

garak：大型语言模型安全探测框架 cs.CL

https://garak.ai

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.11036v1) [paper-pdf](http://arxiv.org/pdf/2406.11036v1)

**Authors**: Leon Derczynski, Erick Galinkin, Jeffrey Martin, Subho Majumdar, Nanna Inie

**Abstract**: As Large Language Models (LLMs) are deployed and integrated into thousands of applications, the need for scalable evaluation of how models respond to adversarial attacks grows rapidly. However, LLM security is a moving target: models produce unpredictable output, are constantly updated, and the potential adversary is highly diverse: anyone with access to the internet and a decent command of natural language. Further, what constitutes a security weak in one context may not be an issue in a different context; one-fits-all guardrails remain theoretical. In this paper, we argue that it is time to rethink what constitutes ``LLM security'', and pursue a holistic approach to LLM security evaluation, where exploration and discovery of issues are central. To this end, this paper introduces garak (Generative AI Red-teaming and Assessment Kit), a framework which can be used to discover and identify vulnerabilities in a target LLM or dialog system. garak probes an LLM in a structured fashion to discover potential vulnerabilities. The outputs of the framework describe a target model's weaknesses, contribute to an informed discussion of what composes vulnerabilities in unique contexts, and can inform alignment and policy discussions for LLM deployment.

摘要: 随着大型语言模型(LLM)的部署和集成到数以千计的应用程序中，对模型如何响应对手攻击的可扩展评估的需求迅速增长。然而，LLM安全是一个不断变化的目标：模型产生不可预测的输出，不断更新，潜在对手高度多样化：任何人都可以访问互联网，并相当熟练地掌握自然语言。此外，在一种情况下，什么构成安全薄弱，在另一种情况下可能不是问题；一刀切的护栏仍然是理论上的。在这篇文章中，我们认为现在是时候重新思考什么是“LLM安全”，并追求一种全面的方法来进行LLM安全评估，其中探索和发现问题是核心。为此，本文介绍了GARAK(生成性人工智能红团队和评估工具包)，这是一个可以用来发现和识别目标LLM或对话系统中的漏洞的框架。Garak以结构化方式探测LLM，以发现潜在漏洞。该框架的输出描述了目标模型的弱点，有助于对在特定环境中构成漏洞的因素进行明智的讨论，并可以为LLM部署的调整和策略讨论提供信息。



## **45. Threat Modelling and Risk Analysis for Large Language Model (LLM)-Powered Applications**

大型语言模型（LLM）支持的应用程序的威胁建模和风险分析 cs.CR

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.11007v1) [paper-pdf](http://arxiv.org/pdf/2406.11007v1)

**Authors**: Stephen Burabari Tete

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized various applications by providing advanced natural language processing capabilities. However, this innovation introduces new cybersecurity challenges. This paper explores the threat modeling and risk analysis specifically tailored for LLM-powered applications. Focusing on potential attacks like data poisoning, prompt injection, SQL injection, jailbreaking, and compositional injection, we assess their impact on security and propose mitigation strategies. We introduce a framework combining STRIDE and DREAD methodologies for proactive threat identification and risk assessment. Furthermore, we examine the feasibility of an end-to-end threat model through a case study of a custom-built LLM-powered application. This model follows Shostack's Four Question Framework, adjusted for the unique threats LLMs present. Our goal is to propose measures that enhance the security of these powerful AI tools, thwarting attacks, and ensuring the reliability and integrity of LLM-integrated systems.

摘要: 大型语言模型(LLM)的出现提供了先进的自然语言处理能力，使各种应用发生了革命性的变化。然而，这一创新带来了新的网络安全挑战。本文探讨了专门为LLM支持的应用程序量身定做的威胁建模和风险分析。针对数据中毒、快速注入、SQL注入、越狱、成分注入等潜在攻击，评估了它们对安全的影响，并提出了缓解策略。我们引入了一个结合STRIDE和DREAD方法的框架，用于主动识别威胁和风险评估。此外，我们还通过一个定制的基于LLM的应用程序的案例研究，研究了端到端威胁模型的可行性。该模型遵循ShoStack的四个问题框架，针对LLMS存在的独特威胁进行了调整。我们的目标是提出措施，增强这些强大的人工智能工具的安全性，挫败攻击，并确保LLM集成系统的可靠性和完整性。



## **46. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2404.01318v3) [paper-pdf](http://arxiv.org/pdf/2404.01318v3)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源的基准测试，包括以下组件：(1)一个不断发展的最新对手提示库，我们称之为越狱人工制品；(2)一个包含100种行为的越狱数据集--既有原始的，也有源自以前工作的--与OpenAI的使用策略保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **47. ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator**

ATM：对抗性调整多代理系统打造强大的检索增强生成器 cs.CL

18 pages, 7 figures

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2405.18111v2) [paper-pdf](http://arxiv.org/pdf/2405.18111v2)

**Authors**: Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, Lei Sha

**Abstract**: Large language models (LLMs) are proven to benefit a lot from retrieval-augmented generation (RAG) in alleviating hallucinations confronted with knowledge-intensive questions. RAG adopts information retrieval techniques to inject external knowledge from semantic-relevant documents as input contexts. However, due to today's Internet being flooded with numerous noisy and fabricating content, it is inevitable that RAG systems are vulnerable to these noises and prone to respond incorrectly. To this end, we propose to optimize the retrieval-augmented Generator with a Adversarial Tuning Multi-agent system (ATM). The ATM steers the Generator to have a robust perspective of useful documents for question answering with the help of an auxiliary Attacker agent. The Generator and the Attacker are tuned adversarially for several iterations. After rounds of multi-agent iterative tuning, the Generator can eventually better discriminate useful documents amongst fabrications. The experimental results verify the effectiveness of ATM and we also observe that the Generator can achieve better performance compared to state-of-the-art baselines.

摘要: 事实证明，大型语言模型(LLM)在缓解面对知识密集型问题时的幻觉方面，从检索增强生成(RAG)中受益匪浅。RAG采用信息检索技术，从与语义相关的文档中注入外部知识作为输入上下文。然而，由于当今的互联网充斥着大量噪声和捏造的内容，RAG系统不可避免地容易受到这些噪声的影响，并容易做出错误的响应。为此，我们提出了用对抗性调谐多智能体系统(ATM)来优化检索增强生成器。ATM引导生成器在辅助攻击者代理的帮助下具有用于问题回答的有用文档的健壮视角。生成器和攻击者被敌对地调整了几次迭代。经过几轮多代理迭代调整后，Generator最终可以更好地区分有用的文档和捏造的文档。实验结果验证了ATM的有效性，并且我们还观察到，与最先进的基线相比，该生成器可以获得更好的性能。



## **48. RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models**

RWKU：大型语言模型的现实世界知识学习基准 cs.CL

48 pages, 7 figures, 12 tables

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10890v1) [paper-pdf](http://arxiv.org/pdf/2406.10890v1)

**Authors**: Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: Large language models (LLMs) inevitably memorize sensitive, copyrighted, and harmful knowledge from the training corpus; therefore, it is crucial to erase this knowledge from the models. Machine unlearning is a promising solution for efficiently removing specific knowledge by post hoc modifying models. In this paper, we propose a Real-World Knowledge Unlearning benchmark (RWKU) for LLM unlearning. RWKU is designed based on the following three key factors: (1) For the task setting, we consider a more practical and challenging unlearning setting, where neither the forget corpus nor the retain corpus is accessible. (2) For the knowledge source, we choose 200 real-world famous people as the unlearning targets and show that such popular knowledge is widely present in various LLMs. (3) For the evaluation framework, we design the forget set and the retain set to evaluate the model's capabilities across various real-world applications. Regarding the forget set, we provide four four membership inference attack (MIA) methods and nine kinds of adversarial attack probes to rigorously test unlearning efficacy. Regarding the retain set, we assess locality and utility in terms of neighbor perturbation, general ability, reasoning ability, truthfulness, factuality, and fluency. We conduct extensive experiments across two unlearning scenarios, two models and six baseline methods and obtain some meaningful findings. We release our benchmark and code publicly at http://rwku-bench.github.io for future work.

摘要: 大型语言模型不可避免地会记住来自训练语料库的敏感、受版权保护和有害的知识；因此，从模型中删除这些知识至关重要。机器遗忘是通过事后修改模型来有效去除特定知识的一种很有前途的解决方案。本文提出了一种用于LLM遗忘的真实世界知识遗忘基准(RWKU)。RWKU的设计基于以下三个关键因素：(1)对于任务设置，我们考虑了一个更实际和更具挑战性的遗忘环境，其中忘记语料库和保留语料库都是不可访问的。(2)在知识源方面，我们选择了200名现实世界名人作为遗忘对象，发现这些流行知识广泛存在于各种学习记忆中。(3)对于评估框架，我们设计了遗忘集和保留集来评估模型在各种实际应用中的能力。对于遗忘集，我们提供了四种成员推理攻击(MIA)方法和九种对抗性攻击探头来严格测试遗忘效果。对于保留集，我们根据邻域扰动、一般能力、推理能力、真实性、真实性和流畅性来评估局部性和效用。我们在两个遗忘场景、两个模型和六个基线方法上进行了广泛的实验，并获得了一些有意义的发现。我们在http://rwku-bench.github.io上公开发布了我们的基准测试和代码，以备将来的工作使用。



## **49. KGPA: Robustness Evaluation for Large Language Models via Cross-Domain Knowledge Graphs**

KGMA：通过跨领域知识图对大型语言模型进行稳健性评估 cs.CL

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10802v1) [paper-pdf](http://arxiv.org/pdf/2406.10802v1)

**Authors**: Aihua Pei, Zehua Yang, Shunan Zhu, Ruoxi Cheng, Ju Jia, Lina Wang

**Abstract**: Existing frameworks for assessing robustness of large language models (LLMs) overly depend on specific benchmarks, increasing costs and failing to evaluate performance of LLMs in professional domains due to dataset limitations. This paper proposes a framework that systematically evaluates the robustness of LLMs under adversarial attack scenarios by leveraging knowledge graphs (KGs). Our framework generates original prompts from the triplets of knowledge graphs and creates adversarial prompts by poisoning, assessing the robustness of LLMs through the results of these adversarial attacks. We systematically evaluate the effectiveness of this framework and its modules. Experiments show that adversarial robustness of the ChatGPT family ranks as GPT-4-turbo > GPT-4o > GPT-3.5-turbo, and the robustness of large language models is influenced by the professional domains in which they operate.

摘要: 用于评估大型语言模型（LLM）稳健性的现有框架过度依赖特定的基准，增加了成本，并且由于数据集限制而无法评估LLM在专业领域的性能。本文提出了一个框架，该框架通过利用知识图（KG）系统评估LLM在对抗性攻击场景下的稳健性。我们的框架从知识图的三重组中生成原始提示，并通过中毒创建对抗提示，通过这些对抗攻击的结果评估LLM的稳健性。我们系统地评估该框架及其模块的有效性。实验表明，ChatGPT家族的对抗鲁棒性排名为GPT-4-涡轮> GPT-4 o> GPT-3.5-涡轮，大型语言模型的鲁棒性受到其运行的专业领域的影响。



## **50. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

了解LLC中的越狱攻击：表示空间分析 cs.CL

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.10794v1) [paper-pdf](http://arxiv.org/pdf/2406.10794v1)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.

摘要: 大型语言模型（LLM）容易受到一种称为越狱的攻击，这种攻击会误导LLM输出有害内容。尽管越狱攻击策略多种多样，但对于为什么有些方法成功而另一些方法失败，人们并没有统一的理解。本文探讨了LLM表示空间中有害和无害提示的行为，以研究成功越狱攻击的内在属性。我们假设成功的攻击具有一些相似的属性：它们有效地将有害提示的表示移向无害提示的方向。我们将隐藏的表示利用到现有越狱攻击的目标中，以沿着接受方向移动攻击，并使用提出的目标进行实验来验证上述假设。我们希望这项研究为理解LLM如何理解有害信息提供新的见解。



