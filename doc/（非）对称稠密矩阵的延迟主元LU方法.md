[TOC]

### 非对称稠密矩阵的延迟主元LU方法

- 在超节点方法中，我们会针对单个超节点的主元块将其作为一个稠密矩阵进行LU分解，按照列主元的方法，我们针对每一列进行选主元，然后更新。若是出现某个列待分解的部分全0，那么我们就选不出列主元，此时Lapack中dgetrf函数的做法是先跳过此列，继续进行下一列的分解，最后报错，这样的做法显然不符合我们最终求解的需要。
- 一个子主元块无法分解，不代表全部的矩阵无法分解。我们可以把无法分解的部分传递给父节点再尝试分解。

<div align="center">
<table style="border-collapse: collapse; table-layout: fixed; width: 300px; height: 300px; text-align: center;">
  <tr>
    <td style="border: 1px solid red; width: 180px; height: 180px;">Fully Summed</td>
    <td style="border: 1px solid;">U Factor</td>
  </tr>
  <tr>
    <td style="border: 1px solid; width: 180px; height: 120px;">L Factor</td>
    <td style="border: 1px solid;">CB Block</td>
  </tr>
</table>

- 我们需要处理如上图所示的波前矩阵:
  - Fully-Summed代表着已经装配好可以进行分解的矩阵，即为超节点的对角块；
  - L/U Factor是我们需要根据Fully-Summed部分的分解结果更新的块，作为最终的结果呈现；
  - CB Block是子节点对于父节点的贡献快，需要在更新之后传递给父节点。
- 注意到对角块之外的行列来自于和对角块中行列耦合的行列。
- 在此方法中，我们仍采用right-looking的列主元方法+KJI更新方法。

#### Step1 分解Fully Summed块

- 我们首先单独考虑对Fully Summed块的分解。
- 红色部分的块表示已经分解好的部分，接下来需要对Unfactored Block进行分解 ，一个比较自然的思路是：
  - 首先进行选列主元，如果能选到合适的列主元，则进行行交换P，将其作为主元计算；
  - 若不能选取到合适的列主元，那么我们就针对全局选取主元，此时需要做行列交换，将右下角的全局最大主元置换到P的位置，然后根据P的值计算$L_p$，然后更新$U_p$和右下角块。


<div style="
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 30px;
">
<!-- ---------------------------------------------------------------- -->  
<div align="center">
<table style="border-collapse: collapse; table-layout: fixed; width: 300px; height: 300px; text-align: center;">
  <tr>
    <td style="border: 1px solid red; width: 100px; height: 100px; padding: 0;">
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" style="display:block">
        <line x1="0" y1="0"
              x2="100" y2="100"
              stroke="red"
              stroke-width="1"
              stroke-dasharray="5,5"/>
        <text x="70" y="35"
          text-anchor="middle">
            U<tspan baseline-shift="sub">EE</tspan>
    	</text>
        <text x="33" y="75"
          text-anchor="middle">
 		  	L<tspan baseline-shift="sub">EE</tspan>
		</text>
      </svg>
    </td>
      <td style="border: 1px solid red;">U<sub>ED</sub></td>
      </tr>
  <tr>
    <td style="border: 1px solid red; width: 100px; height: 200px;">L<sub>DE</sub></td>
    <td style="border: 1px solid;">Unfactored Block</td>
  </tr>
</table>
</div>
<!-- ---------------------------------------------------------------- -->  
<div align="center">
<table style="border-collapse: collapse; table-layout: fixed; width: 300px; height: 300px; text-align: center;">
  <tr>
    <td style="border: 1px solid red; width: 100px; height: 100px; padding: 0;">
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" style="display:block">
        <line x1="0" y1="0"
              x2="100" y2="100"
              stroke="red"
              stroke-width="1"
              stroke-dasharray="5,5"/>
        <text x="70" y="35"
          text-anchor="middle">
            U<tspan baseline-shift="sub">EE</tspan>
    	</text>
        <text x="33" y="75"
          text-anchor="middle">
 		  	L<tspan baseline-shift="sub">EE</tspan>
		</text>
      </svg>
    </td>
      <td style="border: 1px solid red;width: 200px; height: 100px;">U<sub>ED</sub></td>
      </tr>
  <tr>
    <td style="border: 1px solid red; width: 100px; height: 200px;">L<sub>DE</sub></td>
    <td style="border: 1px solid; width: 200px; height: 200px; padding: 0;">
      <table style="border-collapse: collapse; width: 100%; height: 100%; text-align: center;">
        <tr style="height: 20px;">
          <td style="border-right: 1px solid; border-bottom: 1px solid; width: 20px;">P</td>
          <td style="border-bottom: 1px solid;">U<sub>P</sub></td>
        </tr>
        <tr>
          <td style="border-right: 1px solid;">L<sub>P</sub></td>
          <td>......</td>
        </tr>
      </table>
  </tr>
</table>
</div>
<!-- ---------------------------------------------------------------- -->  
</div>

- 设第$k$步尚未分解的活动子块为：

$$
S^{(k)}=\left[s_{ij}^{(k)}\right]_{i,j=k}^{f},
$$

- 其中$f$为Fully Summed块的阶数，$\tau\in(0,1)$为主元接受阈值。

**列主元：**只搜索活动子块的第$k$列，候选主元所在行为
$$
p_k=\underset{k\le i\le f}{\operatorname{argmax}}
      \left|s_{ik}^{(k)}\right|,
\qquad
\alpha_k=\left|s_{p_k k}^{(k)}\right|.
$$

- 记活动子块中的最大元素为：

$$
\beta_k=\left\|S^{(k)}\right\|_{\max}
=\max_{k\le i,j\le f}\left|s_{ij}^{(k)}\right|.
$$

当 $\alpha_k\ge \tau\beta_k$ 时，认为列主元足够稳定。用行置换矩阵$P_k$交换第$k$行和第$p_k$行：
$$
\widetilde S^{(k)}=P_kS^{(k)},
\qquad
d_k=\widetilde s_{kk}^{(k)}=s_{p_k k}^{(k)}.
$$

**全主元：**若$\alpha_k<\tau\beta_k$，则在整个活动子块中搜索最大元素：
$$
(p_k,q_k)=\underset{k\le i,j\le f}{\operatorname{argmax}}
           \left|s_{ij}^{(k)}\right|.
$$

分别用$P_k$和$Q_k$将第$p_k$行、第$q_k$列交换到第$k$行、第$k$列：
$$
\widetilde S^{(k)}=P_kS^{(k)}Q_k,
\qquad
d_k=\widetilde s_{kk}^{(k)}=s_{p_kq_k}^{(k)},
\qquad
|d_k|=\beta_k.
$$

因此，只使用列主元时最终满足$PA=LU$；使用过全主元时则应记录行、列两种置换，最终满足$PAQ=LU.$

**更新：**主元$d_k$确定后，对主元行、主元列和右下角剩余子块进行right-looking更新：
$$
(L_p)_i=\frac{\widetilde s_{ik}^{(k)}}{d_k},
\quad k<i\le f,
\qquad
(U_p)_j=\widetilde s_{kj}^{(k)},
\quad k<j\le f,
$$

$$
s_{ij}^{(k+1)}
=\widetilde s_{ij}^{(k)}-(L_p)_i(U_p)_j
=\widetilde s_{ij}^{(k)}
-\frac{\widetilde s_{ik}^{(k)}\widetilde s_{kj}^{(k)}}{d_k},
\qquad k<i,j\le f.
$$

- 若$\beta_k=0$，则$S^{(k)}$为全零块，列主元和全主元都不存在，在数值意义下不可继续分解，需要将未分解变量延迟到父节点。

- 完成上述步骤后$P$、$U_p$、$L_p$会被归于已分解部分，剩下的作为Unfactored Block进行下一步迭代。

<div align="center">
<table style="border-collapse: collapse; table-layout: fixed; width: 300px; height: 300px; text-align: center;">
  <tr>
    <td style="border: 1px solid red; width: 240px; height: 240px; padding: 0;">
      <svg width="100%" height="100%" viewBox="0 0 240 240" preserveAspectRatio="none" style="display:block">
        <line x1="0" y1="0"
              x2="240" y2="240"
              stroke="red"
              stroke-width="1"
              stroke-dasharray="5,5"/>
        <text x="160" y="85"
      		text-anchor="middle"
     	  	font-size="15">
  				U<tspan baseline-shift="sub">EE</tspan>
				</text>
				<text x="70" y="165"
    		  text-anchor="middle"
    		  font-size="15">
 				  L<tspan baseline-shift="sub">EE</tspan>
				</text>		
      </svg>
    </td>
      <td style="border: 1px solid red;">U<sub>ED</sub></td>
      </tr>
  <tr>
    <td style="border: 1px solid red; width: 240px; height: 60px;">L<sub>DE</sub></td>
    <td style="border: 1px solid;">0</td>
  </tr>
</table>	
</div>

- 实际上若是完全采用上面的做法，那么算法将会消耗掉巨量的时间在搜索全部元与更新上，于是我们提出以下两种优化方法：

**列延迟：**仔细思考全主元算法，我们发现：利用行列变换将全局最大值替换到我们正在进行分解的第k个主元，本质上是将全局最大值那一列通过列交换交换到了第k列，然后进行了一次列主元分解，而本身的第k列因为已经选不出全局主元，故第k列一定是全0的，那么在之后的分解过程中，全零列既不会被选出全主元来，也不会被更新。原因是被放到右边的列在当前活动部分全为零：\[ A(k:m-1,j)=0. \]之后的 LU 更新为\[ A(k+1:m-1,j) \leftarrow A(k+1:m-1,j)-L(:,k)U(k,j). \]因为 \[ U(k,j)=0, \]所以更新项也是零。所以选全主元的步骤可以被替换为以下列延迟步骤：

- 若$\alpha_k<\tau\beta_k$，则我们将该列直接移动到矩阵的 Unfactored Block 的尾部去，并记录这个交换的步骤。
- 为了保证之后的列进行延迟被移动到尾部的时候不会再次将本来就被存放在Unfactored Block 的尾部的零列给置换到前方去，从而导致死循环，我们可以通过以下算法避免问题：
  - 从右向左找第一列可用列，找到后立即停止，不比较其他列。
  - 在该列内部使用列主元法，并将其和当前的零列替换位置。‘
  - 最后如果我们搜索到当前列都找不到可用列，那么说明所有的Unfactored Block已经全零，剩下的所有主元全部延迟。

**更细粒度的自适应KJI-SAXPY方法优化：**在Lapack的dgetrf中，采用到了类似的优化方法（之后的step2也是这个思路）：我们如果每次选主元后立刻更新所有项目，那么显然是会很慢的，我们可以借助blas的三角求解和矩阵乘法针对矩阵块进行快速更新，这就需要我们在分解一部分主元之后一起更新。

- demo中选取panel长度为64，然后在这个panel中，我们选列主元（注意不是在64*64的panel块中选，而是在当前列的所有活动列中选取，所以说panel其实是一个长方形矩阵）后只更新当前长方形panel中的数，只需要保证选主元的时候当前列已经被更新就行了，当选完一个panel之后，我们在更新剩下的部分，然后再选取新的panel进行更新
- 但是我们的流程中会遇到延迟主元的情况，此时会对当前panel产生影响，为了保证数值分解的正确性，我们让算法在遇到延迟主元的时候立刻停止当前panel的分解，并按照已经分解的部分将矩阵的剩余部分全部更新一次，然后再从被换过来的非零列开始新一个panel的分解。这样既做到了利用blas对整个分解过程加速，也保证了数值分解的正确性。

- 经过测试后，采用列延迟的方法大概可以将整个计算过程时间缩短90%，而采用更细粒度的自适应KJI-SAXPY方法优化后速度还能提升一倍，整体优化后的算子能力在恶意设计过的算例上的时间已经略快于Lapack。

#### Step2 更新波前矩阵

- 此时波前矩阵的Fully-Summed的部分已经被分成了5个部分，我们根据已经分解完毕的情况对其他部分进行更新：

<div style="
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 30px;
">
<!-- ---------------------------------------------------------------- -->  
<div align="center">
<table style="border-collapse: collapse; table-layout: fixed; width: 300px; height: 300px; text-align: center;">
  <tr>
    <td style="border: 1px solid red; width: 150px; height: 150px; padding: 0;">
      <svg width="100%" height="100%" viewBox="0 0 150 150" preserveAspectRatio="none" style="display:block">
        <line x1="0" y1="0"
              x2="150" y2="150"
              stroke="red"
              stroke-width="1"
              stroke-dasharray="5,5"/>
        <text x="110" y="55"
          text-anchor="middle">
            U<tspan baseline-shift="sub">EE</tspan>
    	</text>
        <text x="45" y="105"
          text-anchor="middle">
 		  	L<tspan baseline-shift="sub">EE</tspan>
		</text>
      </svg>
    </td>
      <td style="border: 1px solid red; width: 60px; height: 150px;">U<sub>ED</sub></td>
    	<td style="border: 1px solid;">A<sub>EU</sub></td>
      </tr>
  <tr>
    <td style="border: 1px solid red; width: 150px; height: 60px;">L<sub>DE</sub></td>
    <td style="border: 1px solid red;">0</td>
    <td style="border: 1px solid;">A<sub>DU</sub></td>
  </tr>
  <tr>
    <td style="border: 1px solid; width: 150px; height: 90px;">A<sub>UE</sub></td>
    <td style="border: 1px solid;">A<sub>UD</sub></td>
    <td style="border: 1px solid;">A<sub>UU</sub></td>
  </tr>
</table>
</div>
<!-- ---------------------------------------------------------------- -->  
<div align="center">
<table style="border-collapse: collapse; table-layout: fixed; width: 300px; height: 300px; text-align: center;">
  <tr>
    <td style="border: 1px solid; width: 150px; height: 150px; padding: 0;">
      <svg width="100%" height="100%" viewBox="0 0 150 150" preserveAspectRatio="none" style="display:block">
        <line x1="0" y1="0"
              x2="150" y2="150"
              stroke="black"
              stroke-width="1"
              stroke-dasharray="5,5"/>
        <text x="110" y="55"
          text-anchor="middle">
            U<tspan baseline-shift="sub">EE</tspan>
    	</text>
        <text x="45" y="105"
          text-anchor="middle">
 		  	L<tspan baseline-shift="sub">EE</tspan>
		</text>
      </svg>
    </td>
      <td style="border: 1px solid; border-bottom-color: red; width: 60px; height: 150px;">U<sub>ED</sub></td>
	    	<td style="border: 1px solid; border-bottom-color: red;">U<sub>EU</sub></td>
      </tr>
  <tr>
    <td style="border: 1px solid; border-right-color: red; width: 150px; height: 60px;">L<sub>DE</sub></td>
    <td style="border: 1px solid red;">0</td>
    <td style="border: 1px solid red;">A<sub>DU</sub></td>
  </tr>
  <tr>
    <td style="border: 1px solid; border-right-color: red; width: 150px; height: 90px;">L<sub>UE</sub></td>
    <td style="border: 1px solid red;">A<sub>UD</sub></td>
    <td style="border: 1px solid red;">A<sub>UU</sub></td>
  </tr>
</table>
</div>
<!-- ---------------------------------------------------------------- -->  
</div>

**Step2.1 更新 $A_{UE}、A_{EU}$ 得到 $L_{UE}、U_{EU}$ ：**

- 利用三角求解计算：

$$
U_{EU}=L_{EE}^{-1}A_{EU}\quad L_{UE}=A_{UE}U_{EE}^{-1}
$$
**Step2.2 更新 $A_{DU}、A_{UD}$：**
$$
A_{DU} \leftarrow A_{DU}-L_{DE}U_{EU}\\
\\
A_{UD} \leftarrow A_{UD}-L_{UE}U_{ED}
$$
**Step2.3 更新 $A_{UU}$：** 
$$
A_{UU} \leftarrow A_{UU}-L_{UE}U_{EU}
$$

#### Step3 传递贡献块给父节点

- 在Step2中我们更新了所有的部分，并将右下的四个子矩阵组成扩展贡献块传递给父节点组装。
- 这里的“组装”不是把整个稠密块按几何位置直接拼到父矩阵上，而是根据贡献块的**全局行列编号**做 `extend-add`（扩展—累加）。
- 普通更新变量与父节点已有索引匹配后累加，子节点未消去的延迟变量则动态扩充父节点的候选主元区域。

<div align="center">
<svg width="760" height="620" viewBox="0 0 760 620"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="延迟主元贡献块向父节点传递示意图"
     style="max-width: 100%; height: auto; display: block;">
  <!-- 整个父节点的波前矩阵 -->
  <rect x="20" y="20" width="720" height="580"
        fill="none" stroke="black" stroke-width="2"/>
  <!-- 紧凑表示已处理的超节点 S1, ..., Sn -->
  <rect x="20" y="20" width="42" height="42"
        fill="white" stroke="black" stroke-width="2"/>
  <text x="41" y="47" text-anchor="middle" font-size="17">S1</text>
  <text x="86" y="88" text-anchor="middle" font-size="22">…</text>
  <rect x="110" y="95" width="42" height="42"
        fill="white" stroke="black" stroke-width="2"/>
  <text x="131" y="122" text-anchor="middle" font-size="17">Sn</text>
  <!-- S1 的行、列边界向大矩阵边缘延伸 -->
  <path d="M62 62 H740 M62 62 V600"
        fill="none" stroke="black" stroke-width="1.5"
        stroke-dasharray="7 8"/>
  <!-- Sn 左边向下、上边向右延伸 -->
  <path d="M110 137 V600 M152 95 H740"
        fill="none" stroke="black" stroke-width="1.5"
        stroke-dasharray="7 8"/>
  <!-- 右下角四个分块的上边和左边 -->
  <path d="M152 137 H740 M152 137 V600"
        fill="none" stroke="black" stroke-width="2"/>
  <!-- 红色线：扩展后贡献块的上、左边界 -->
  <path d="M520 20 V420 M20 420 H520"
        fill="none" stroke="#e02020" stroke-width="3"/>
  <path d="M520 137 V420 M152 420 H520"
        fill="none" stroke="#e02020" stroke-width="3"/>
  <!-- 其余分块边界 -->
  <path d="M520 420 V600 M520 420 H740"
        fill="none" stroke="black" stroke-width="2"/>
  <!-- 分块名称 -->
  <text x="336" y="276" text-anchor="middle" font-size="22">Fully Summed</text>
  <text x="336" y="306" text-anchor="middle" font-size="18">Eₚ ∪ Dᶜ</text>
  <text x="630" y="276" text-anchor="middle" font-size="25">U Factor</text>
  <text x="630" y="306" text-anchor="middle" font-size="18">Uₚ</text>
  <text x="336" y="510" text-anchor="middle" font-size="25">L Factor</text>
  <text x="336" y="540" text-anchor="middle" font-size="18">Eₚ ∪ Dᶜ</text>
  <text x="630" y="510" text-anchor="middle" font-size="25">CB Block</text>
</svg>
</div>
##### Step3.1 子节点传出的内容

- 设子节点已消去的变量为 $E_c$，未能消去的延迟变量为 $D_c$，原来的普通更新变量为 $U_c$。子节点传出的扩展贡献块为

$$
C_c=
\begin{bmatrix}
S_{DD} & S_{DU}\\
S_{UD} & S_{UU}
\end{bmatrix},
\qquad I_c^{\mathrm{out}}=D_c\cup U_c.
$$

- 除了数值 $C_c$ 以外，还必须同时传递其全局行、列索引数组。对于非对称LU，行索引和列索引应分开保存：

$$
R_c^{\mathrm{out}}=D_c^r\cup U_c^r,
\qquad
C_c^{\mathrm{out}}=D_c^c\cup U_c^c.
$$

- 数值块中的第 $(i,j)$ 个元素，必须与 $R_c^{\mathrm{out}}(i)$ 和 $C_c^{\mathrm{out}}(j)$ 两个全局编号绑定，不能只传递一个没有索引的稠密数组。

##### Step3.2 父节点建立新的索引集

- 设父节点原来的候选主元集为 $E_p$，更新集为 $U_p$。正确的符号分析通常应保证子节点的普通更新变量已在父波前中，即：

$$
U_c\subseteq E_p\cup U_p.
$$

- “普通更新变量”是相对子节点而言的。它们进入父节点后要根据符号分析确定的消去归属再次分类：
  - 其中 $U_c\cap E_p$ 在子节点只是更新变量，但到达父节点后已经Fully Summed，可在父节点参与选主元；
  - $U_c\cap U_p$ 则仍然留在父节点的更新区。

$$
U_c=(U_c\cap E_p)\cup(U_c\cap U_p).
$$

- 延迟变量 $D_c$ 是数值分解的过程中动态产生的部分，因此父节点需要将其加入新的候选主元集：

$$
E_p^{\mathrm{new}}=E_p\cup D_c,
\qquad
I_p^{\mathrm{new}}=E_p^{\mathrm{new}}\cup U_p.
$$

##### Step3.3 建立局部映射并累加数值

- 父节点为每个全局行列编号建立到局部存储位置的映射：

$$
\operatorname{map}_r(g)=g\text{在父波前行索引中的局部位置},
$$

$$
\operatorname{map}_c(g)=g\text{在父波前列索引中的局部位置}.
$$

- 若扩展后出现新的行或列，则先扩展父波前的存储空间，并将新位置初始化为零。然后对子贡献块执行散射累加：

$$
F_p\!\left(\operatorname{map}_r(R_c^{\mathrm{out}}(i)),
             \operatorname{map}_c(C_c^{\mathrm{out}}(j))\right)
\mathrel{+}=C_c(i,j).
$$

- 父矩阵中原来没有数值的位置按零处理，累加子节点贡献后就成为新的fill-in。如果多个子节点对同一个全局位置有贡献，则必须将它们全部累加，而不是相互覆盖。

##### Step3.4 组装后的分区与继续传递

- 完成所有子贡献块的 `extend-add` 之后，父节点按下列规则进行操作：

  - $E_p\cup D_c$：放入Fully Summed候选主元区，在父节点重新进行主元检验；
  - 重复step1、step2在父节点成功消去的延迟变量进入 $L/U$ 因子；
  - 仍无法通过主元检验的变量再次进入父节点的扩展贡献块，继续向更高层节点传递。

### 对称稠密矩阵的Bunch-Kaufman方法

- 如果整个波前矩阵是对称的，则Fully-Summed块也是对称的，此时可采用Bunch-Kaufman方法进行带对称主元选取的$LDL^T$分解：

$$
P^T AP=LDL^T.
$$

- 其中$P$是用于同时置换行、列的置换矩阵，$L$为单位下三角矩阵，$D$为由$1\times1$和$2\times2$主元块组成的块对角矩阵。与非对称LU不同，对称分解中不能只交换行而不交换列；交换第$i$、$j$行时，必须同时交换第$i$、$j$列。

<div align="center">
<table style="border-collapse: collapse; table-layout: fixed; width: 300px; height: 300px; text-align: center;">
  <tr>
    <td style="border: 1px solid red; width: 180px; height: 180px;">Fully Summed</td>
    <td style="border: 1px solid;">L<sup>T</sup> Factor</td>
  </tr>
  <tr>
    <td style="border: 1px solid; width: 180px; height: 120px;">L Factor</td>
    <td style="border: 1px solid;">CB Block</td>
  </tr>
</table>
</div>

#### 对称Step1：分解Fully-Summed块

- 设第$k$步尚未分解的活动对称子块为

$$
S^{(k)}=\left[s_{ij}^{(k)}\right]_{i,j=k}^{f},
\qquad S^{(k)}=\left(S^{(k)}\right)^T,
$$

- 其中$f$是当前仍可参与主元选取的Fully-Summed变量个数。Bunch-Kaufman方法在每一步选择一个$1\times1$或$2\times2$对角主元块，并保持后续更新仍然对称。

##### Step1.1 Bunch-Kaufman主元搜索

定义Bunch-Kaufman阈值
$$
\gamma=\frac{1+\sqrt{17}}{8}\approx0.640388.
$$

- 首先检查第$k$列。记当前对角元大小为

$$
a_k=\left|s_{kk}^{(k)}\right|,
$$

- 第$k$列对角线以下的最大元素所在行为

$$
p=\underset{k<i\le f}{\operatorname{argmax}}
   \left|s_{ik}^{(k)}\right|,
\qquad
\lambda_k=\max_{k<i\le f}\left|s_{ik}^{(k)}\right|
         =\left|s_{pk}^{(k)}\right|.
$$

- 若该列不为数值零，则按以下规则选择主元。

1. 若$a_k\ge\gamma\lambda_k$，则直接采用$s_{kk}^{(k)}$作为$1\times1$主元，不需要交换。

2. 若$a_k<\gamma\lambda_k$，则检查候选行$p$。定义该行除对角元之外的最大元素：

$$
\sigma_p=
\max_{\substack{k\le j\le f\\j\ne p}}
\left|s_{pj}^{(k)}\right|.
$$

   因为$|s_{pk}^{(k)}|=\lambda_k$，所以当$\lambda_k>0$时有$\sigma_p\ge\lambda_k>0$。接下来依次判断：

   - 若$a_k\ge\gamma\frac{\lambda_k^2}{\sigma_p}$， 仍使用$s_{kk}^{(k)}$作为$1\times1$主元；

   - 否则，若$\left|s_{pp}^{(k)}\right|\ge\gamma\sigma_p$， 则同时交换第$k$、$p$行和第$k$、$p$列，将$s_{pp}^{(k)}$移到$(k,k)$位置，并将其作为$1\times1$主元；

   - 若上述两个条件都不满足，则选择由第$k$、$p$个变量组成的$2\times2$主元块。通过对称交换将变量$p$移动到$k+1$位置，得到：

$$
D_k=
\begin{bmatrix}
\widetilde s_{kk}^{(k)}&
\widetilde s_{k,k+1}^{(k)}\\
\widetilde s_{k+1,k}^{(k)}&
\widetilde s_{k+1,k+1}^{(k)}
\end{bmatrix}.
$$

##### Step1.2 对称置换

- 对于$1\times1$主元$s_{pp}^{(k)}$，用置换矩阵$P_k$同时交换第$k$、$p$行和第$k$、$p$列：

$$
\widetilde S^{(k)}=P_k^T S^{(k)}P_k,
\qquad
D_k=\left[\widetilde s_{kk}^{(k)}\right].
$$

- 对于由$k$、$p$组成的$2\times2$主元，用$P_k$将这两个变量放到活动子块的前两个位置：

$$
\widetilde S^{(k)}=P_k^T S^{(k)}P_k
=
\begin{bmatrix}
D_k&B_k^T\\
B_k&C_k
\end{bmatrix},
\qquad D_k\in\mathbb R^{2\times2}.
$$

- 置换必须同时作用于整个波前矩阵中对应的行和列，并同步更新全局变量编号。多步置换累积后，最终得到

$$
P=P_1P_2\cdots P_t,
\qquad
P^TFP=LDL^T..
$$

##### Step1.3 检查$2\times2$主元的稳定性

- 设

$$
D_k=
\begin{bmatrix}
a&b\\
b&c
\end{bmatrix},
\qquad
\Delta_k=\det(D_k)=ac-b^2.
$$

  当$\Delta_k\ne0$时，

$$
D_k^{-1}
=\frac{1}{\Delta_k}
\begin{bmatrix}
c&-b\\
-b&a
\end{bmatrix}.
$$

- Bunch-Kaufman判据能够控制消元乘子的增长，但计算中仍应避免直接使用数值上近奇异的$2\times2$块，可以进行额外检查：

$$
\operatorname{rcond}(D_k)
=\frac{1}{\|D_k\|\,\|D_k^{-1}\|}
\ge\tau_D.
$$

  若不计算条件数，也可以使用下面的行列式尺度判据做快速筛选（注意到）：

$$
|\Delta_k|
>\varepsilon_{\mathrm{piv}}\|D_k\|^2.
$$

  若$D_k$不满足稳定性要求，则不能直接求$D_k^{-1}$，应将相关变量延迟到父节点，在更大的Fully-Summed候选集中重新选取主元。

##### Step1.4 计算$L$并更新活动子块

- 设本步主元阶数为$r\in\{1,2\}$。完成对称置换后，将子块写成

$$
\widetilde S^{(k)}
=
\begin{bmatrix}
D_k&B_k^T\\
B_k&C_k
\end{bmatrix},
\qquad D_k\in\mathbb R^{r\times r}.
$$

- 当前块列的消元乘子为

$$
L_k=B_kD_k^{-1},
$$

  右下角活动子块通过对称Schur补更新：

$$
S^{(k+r)}
=C_k-L_kD_kL_k^T
=C_k-B_kD_k^{-1}B_k^T.
$$

- 对于$1\times1$主元$D_k=[d_k]$，上述公式退化为

$$
L_k=\frac{B_k}{d_k},
\qquad
S^{(k+1)}
=C_k-\frac{B_kB_k^T}{d_k}.
$$

- 对于$2\times2$主元，使用

$$
L_k
=\frac{1}{\Delta_k}
B_k
\begin{bmatrix}
c&-b\\
-b&a
\end{bmatrix},
$$

  然后执行秩$2$对称更新

$$
S^{(k+2)}=C_k-L_kD_kL_k^T.
$$

- 每成功消去一个$1\times1$主元令$k\leftarrow k+1$；每成功消去一个$2\times2$主元令$k\leftarrow k+2$。未通过主元检验的变量不进入$D$，而是移动到延迟区；若当前节点剩余的Fully-Summed变量都无法形成稳定主元，则停止本节点的数值消元。
- 延迟操作：如果存在找不到列主元的情况（列全0）则将其移到最后一个主元去准备做延迟。

**块分解优化：**和非对称情况一样，我们尝试用块分解的方法来进行优化。

```cpp
[0, panel_begin)       以前面板已经消去
[panel_begin, k)       当前面板已经接受的主元
[k, active_end)        仍可参与主元选择的活动节点
[active_end, n)        已经延迟的节点
```

设当前面板从 \(b=\text{panel\_begin}\) 开始。以前面板产生的 Schur 补已经全部写回 `A`，所以在面板刚开始时：

- \[ A(b:n-1,b:n-1) \]就是最新的尾矩阵

假设当前面板已经接受了 \(t\) 列：\[ L_P=L(:,b:b+t-1), \]对应的1×1和2×2主元组成：\[ D_P. \]

- 定义工作矩阵：\[ W_P=L_PD_P. \] 那么真正的当前 Schur 补是：\[ S = A_{\mathrm{stored}}-L_PD_PL_P^T = A_{\mathrm{stored}}-L_PW_P^T. \]

- 当前准备处理第 \(k\) 列。因为 `A(:,k)` 还没有包含本面板前面 \(t\) 列主元的更新，所以需要计算对当前列的更新之后才能继续选主元

- 按照bunch-kaufman方法，我们首先判定当前列上本来的对角元是否符合阈值要求，若是满足这标记作为主元
  - 若是不满足我们需要在接下来的判定中运用到当前列的最大值所在行的数据，此时需要更新此行
- 当1x1，2x2主元都不满足要求，即需要延迟主元，我们维护一个active_end值记录尾部被延迟的行/列坐标，在每次需要延迟的时候利用这值向左查找找到下一个非零列，之后交换其与被延迟主元即可。

- 此优化要求在每个panel的分解过程中维护一个工作数组 $W_P$：

  - 理论上可以只保存 \(L_P,D_P\)。但每次需要当前列时，都要先计算：\[ D_PL_P(k,:)^T, \]再计算：\[ L_P\left(D_PL_P(k,:)^T\right). \] 而 \(D_P\) 中混合了1×1和2×2块，处理起来更复杂。

  - 预先保存：\[ W_P=L_PD_P \]以后，当前列直接计算：\[ L_PW_P(k,:)^T. \]只需要一次 `dgemv`，并且不必在每次更新时重新遍历1×1/2×2的 \(D\) 块结构。
  - $W_P$的大小为：$panel\_{size}*active\_size$ ，默认的设置下最大值为64*8192，双精度存储下约耗费4MiB。
  - 注意到：工作矩阵也需要随着矩阵在选主元的时候做的行列交换一起交换。
