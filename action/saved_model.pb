ë!
Ö
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018ï

RMSprop/dense_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/dense_4/bias/momentum

1RMSprop/dense_4/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/bias/momentum*
_output_shapes
:*
dtype0

RMSprop/dense_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!RMSprop/dense_4/kernel/momentum

3RMSprop/dense_4/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/kernel/momentum*
_output_shapes

:@*
dtype0

RMSprop/dense_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameRMSprop/dense_3/bias/momentum

1RMSprop/dense_3/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/bias/momentum*
_output_shapes
:@*
dtype0

RMSprop/dense_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*0
shared_name!RMSprop/dense_3/kernel/momentum

3RMSprop/dense_3/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/kernel/momentum*
_output_shapes

:d@*
dtype0

 RMSprop/gru_cell_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*1
shared_name" RMSprop/gru_cell_1/bias/momentum

4RMSprop/gru_cell_1/bias/momentum/Read/ReadVariableOpReadVariableOp RMSprop/gru_cell_1/bias/momentum*
_output_shapes
:	¬*
dtype0
µ
,RMSprop/gru_cell_1/recurrent_kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*=
shared_name.,RMSprop/gru_cell_1/recurrent_kernel/momentum
®
@RMSprop/gru_cell_1/recurrent_kernel/momentum/Read/ReadVariableOpReadVariableOp,RMSprop/gru_cell_1/recurrent_kernel/momentum*
_output_shapes
:	d¬*
dtype0
¡
"RMSprop/gru_cell_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*3
shared_name$"RMSprop/gru_cell_1/kernel/momentum

6RMSprop/gru_cell_1/kernel/momentum/Read/ReadVariableOpReadVariableOp"RMSprop/gru_cell_1/kernel/momentum*
_output_shapes
:	d¬*
dtype0

RMSprop/gru_cell/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*/
shared_name RMSprop/gru_cell/bias/momentum

2RMSprop/gru_cell/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/gru_cell/bias/momentum*
_output_shapes
:	¬*
dtype0
±
*RMSprop/gru_cell/recurrent_kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*;
shared_name,*RMSprop/gru_cell/recurrent_kernel/momentum
ª
>RMSprop/gru_cell/recurrent_kernel/momentum/Read/ReadVariableOpReadVariableOp*RMSprop/gru_cell/recurrent_kernel/momentum*
_output_shapes
:	d¬*
dtype0

 RMSprop/gru_cell/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*1
shared_name" RMSprop/gru_cell/kernel/momentum

4RMSprop/gru_cell/kernel/momentum/Read/ReadVariableOpReadVariableOp RMSprop/gru_cell/kernel/momentum* 
_output_shapes
:
¬*
dtype0
«
'RMSprop/embedding_1/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*8
shared_name)'RMSprop/embedding_1/embeddings/momentum
¤
;RMSprop/embedding_1/embeddings/momentum/Read/ReadVariableOpReadVariableOp'RMSprop/embedding_1/embeddings/momentum*
_output_shapes
:	*
dtype0
§
%RMSprop/embedding/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%RMSprop/embedding/embeddings/momentum
 
9RMSprop/embedding/embeddings/momentum/Read/ReadVariableOpReadVariableOp%RMSprop/embedding/embeddings/momentum*
_output_shapes
:	*
dtype0
«
)RMSprop/batch_normalization/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)RMSprop/batch_normalization/beta/momentum
¤
=RMSprop/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp)RMSprop/batch_normalization/beta/momentum*
_output_shapes	
:*
dtype0
­
*RMSprop/batch_normalization/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*RMSprop/batch_normalization/gamma/momentum
¦
>RMSprop/batch_normalization/gamma/momentum/Read/ReadVariableOpReadVariableOp*RMSprop/batch_normalization/gamma/momentum*
_output_shapes	
:*
dtype0

RMSprop/dense_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/dense_2/bias/momentum

1RMSprop/dense_2/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/bias/momentum*
_output_shapes	
:*
dtype0

RMSprop/dense_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!RMSprop/dense_2/kernel/momentum

3RMSprop/dense_2/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/kernel/momentum*
_output_shapes
:	*
dtype0

RMSprop/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/dense_1/bias/momentum

1RMSprop/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/momentum*
_output_shapes	
:*
dtype0

RMSprop/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*0
shared_name!RMSprop/dense_1/kernel/momentum

3RMSprop/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/momentum*
_output_shapes
:	@*
dtype0

RMSprop/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameRMSprop/dense/bias/momentum

/RMSprop/dense/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/momentum*
_output_shapes
:@*
dtype0

RMSprop/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_nameRMSprop/dense/kernel/momentum

1RMSprop/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/momentum*
_output_shapes

:@*
dtype0

RMSprop/dense_4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_4/bias/rms

,RMSprop/dense_4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/bias/rms*
_output_shapes
:*
dtype0

RMSprop/dense_4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_nameRMSprop/dense_4/kernel/rms

.RMSprop/dense_4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/kernel/rms*
_output_shapes

:@*
dtype0

RMSprop/dense_3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameRMSprop/dense_3/bias/rms

,RMSprop/dense_3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/bias/rms*
_output_shapes
:@*
dtype0

RMSprop/dense_3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*+
shared_nameRMSprop/dense_3/kernel/rms

.RMSprop/dense_3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/kernel/rms*
_output_shapes

:d@*
dtype0

RMSprop/gru_cell_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*,
shared_nameRMSprop/gru_cell_1/bias/rms

/RMSprop/gru_cell_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/gru_cell_1/bias/rms*
_output_shapes
:	¬*
dtype0
«
'RMSprop/gru_cell_1/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*8
shared_name)'RMSprop/gru_cell_1/recurrent_kernel/rms
¤
;RMSprop/gru_cell_1/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp'RMSprop/gru_cell_1/recurrent_kernel/rms*
_output_shapes
:	d¬*
dtype0

RMSprop/gru_cell_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*.
shared_nameRMSprop/gru_cell_1/kernel/rms

1RMSprop/gru_cell_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/gru_cell_1/kernel/rms*
_output_shapes
:	d¬*
dtype0

RMSprop/gru_cell/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬**
shared_nameRMSprop/gru_cell/bias/rms

-RMSprop/gru_cell/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/gru_cell/bias/rms*
_output_shapes
:	¬*
dtype0
§
%RMSprop/gru_cell/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*6
shared_name'%RMSprop/gru_cell/recurrent_kernel/rms
 
9RMSprop/gru_cell/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp%RMSprop/gru_cell/recurrent_kernel/rms*
_output_shapes
:	d¬*
dtype0

RMSprop/gru_cell/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*,
shared_nameRMSprop/gru_cell/kernel/rms

/RMSprop/gru_cell/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/gru_cell/kernel/rms* 
_output_shapes
:
¬*
dtype0
¡
"RMSprop/embedding_1/embeddings/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"RMSprop/embedding_1/embeddings/rms

6RMSprop/embedding_1/embeddings/rms/Read/ReadVariableOpReadVariableOp"RMSprop/embedding_1/embeddings/rms*
_output_shapes
:	*
dtype0

 RMSprop/embedding/embeddings/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" RMSprop/embedding/embeddings/rms

4RMSprop/embedding/embeddings/rms/Read/ReadVariableOpReadVariableOp RMSprop/embedding/embeddings/rms*
_output_shapes
:	*
dtype0
¡
$RMSprop/batch_normalization/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$RMSprop/batch_normalization/beta/rms

8RMSprop/batch_normalization/beta/rms/Read/ReadVariableOpReadVariableOp$RMSprop/batch_normalization/beta/rms*
_output_shapes	
:*
dtype0
£
%RMSprop/batch_normalization/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%RMSprop/batch_normalization/gamma/rms

9RMSprop/batch_normalization/gamma/rms/Read/ReadVariableOpReadVariableOp%RMSprop/batch_normalization/gamma/rms*
_output_shapes	
:*
dtype0

RMSprop/dense_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_2/bias/rms

,RMSprop/dense_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/bias/rms*
_output_shapes	
:*
dtype0

RMSprop/dense_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameRMSprop/dense_2/kernel/rms

.RMSprop/dense_2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/kernel/rms*
_output_shapes
:	*
dtype0

RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_1/bias/rms

,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes	
:*
dtype0

RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*+
shared_nameRMSprop/dense_1/kernel/rms

.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms*
_output_shapes
:	@*
dtype0

RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:@*
dtype0

RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_nameRMSprop/dense/kernel/rms

,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
_output_shapes

:@*
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:@*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:d@*
dtype0
{
gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬* 
shared_namegru_cell_1/bias
t
#gru_cell_1/bias/Read/ReadVariableOpReadVariableOpgru_cell_1/bias*
_output_shapes
:	¬*
dtype0

gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*,
shared_namegru_cell_1/recurrent_kernel

/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru_cell_1/recurrent_kernel*
_output_shapes
:	d¬*
dtype0

gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*"
shared_namegru_cell_1/kernel
x
%gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpgru_cell_1/kernel*
_output_shapes
:	d¬*
dtype0
w
gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*
shared_namegru_cell/bias
p
!gru_cell/bias/Read/ReadVariableOpReadVariableOpgru_cell/bias*
_output_shapes
:	¬*
dtype0

gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬**
shared_namegru_cell/recurrent_kernel

-gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru_cell/recurrent_kernel*
_output_shapes
:	d¬*
dtype0
|
gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬* 
shared_namegru_cell/kernel
u
#gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru_cell/kernel* 
_output_shapes
:
¬*
dtype0

embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameembedding_1/embeddings

*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	*
dtype0

embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0

NoOpNoOp
æ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 
valueB B
·
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
mlp
	mlp2

emb_act
emb_ind
add

concat
rnn1
rnn2
	q_net
	optimizer
call

signatures*
ª
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21*

0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15
&16
'17
(18
)19*
* 
°
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
/trace_0
0trace_1
1trace_2
2trace_3* 
6
3trace_0
4trace_1
5trace_2
6trace_3* 
* 
Þ
7layer_with_weights-0
7layer-0
8layer_with_weights-1
8layer-1
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
Þ
?layer_with_weights-0
?layer-0
@layer_with_weights-1
@layer-1
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
 
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

embeddings*
 
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

embeddings*

S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 

Y	keras_api* 
Ó
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`_random_generator

 kernel
!recurrent_kernel
"bias*
Ó
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator

#kernel
$recurrent_kernel
%bias*
Þ
hlayer_with_weights-0
hlayer-0
ilayer_with_weights-1
ilayer-1
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses*

piter
	qdecay
rlearning_rate
smomentum
trho
rms
rms
rms
rms
rms
rms 
rms¡
rms¢
rms£
rms¤
 rms¥
!rms¦
"rms§
#rms¨
$rms©
%rmsª
&rms«
'rms¬
(rms­
)rms®momentum¯momentum°momentum±momentum²momentum³momentum´momentumµmomentum¶momentum·momentum¸ momentum¹!momentumº"momentum»#momentum¼$momentum½%momentum¾&momentum¿'momentumÀ(momentumÁ)momentumÂ*
P
utrace_0
vtrace_1
wtrace_2
xtrace_3
ytrace_4
ztrace_5* 

{serving_default* 
LF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEbatch_normalization/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEembedding/embeddings'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEembedding_1/embeddings'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEgru_cell/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEgru_cell/recurrent_kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEgru_cell/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEgru_cell_1/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEgru_cell_1/recurrent_kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEgru_cell_1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_4/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_4/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*

0
1*
C
0
	1

2
3
4
5
6
7
8*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
¨
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*
 
0
1
2
3*
 
0
1
2
3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias*
Ü
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
	¡axis
	gamma
beta
moving_mean
moving_variance*
.
0
1
2
3
4
5*
 
0
1
2
3*
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
:
§trace_0
¨trace_1
©trace_2
ªtrace_3* 
:
«trace_0
¬trace_1
­trace_2
®trace_3* 

0*

0*
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

´trace_0* 

µtrace_0* 

0*

0*
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

»trace_0* 

¼trace_0* 
* 
* 
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

Âtrace_0* 

Ãtrace_0* 
* 

 0
!1
"2*

 0
!1
"2*
* 

Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

Étrace_0
Êtrace_1* 

Ëtrace_0
Ìtrace_1* 
* 

#0
$1
%2*

#0
$1
%2*
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

Òtrace_0
Ótrace_1* 

Ôtrace_0
Õtrace_1* 
* 
¬
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses

&kernel
'bias*
¬
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses

(kernel
)bias*
 
&0
'1
(2
)3*
 
&0
'1
(2
)3*
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*
:
çtrace_0
ètrace_1
étrace_2
êtrace_3* 
:
ëtrace_0
ìtrace_1
ítrace_2
îtrace_3* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ôtrace_0* 

õtrace_0* 

0
1*

0
1*
* 

önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ûtrace_0* 

ütrace_0* 
* 

70
81*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
 
0
1
2
3*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 

0
1*

?0
@1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

&0
'1*

&0
'1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses*

trace_0* 

trace_0* 

(0
)1*

(0
)1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

h0
i1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
vp
VARIABLE_VALUERMSprop/dense/kernel/rmsDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUERMSprop/dense/bias/rmsDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUERMSprop/dense_1/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUERMSprop/dense_1/bias/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUERMSprop/dense_2/kernel/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUERMSprop/dense_2/bias/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE%RMSprop/batch_normalization/gamma/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE$RMSprop/batch_normalization/beta/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE RMSprop/embedding/embeddings/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE"RMSprop/embedding_1/embeddings/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUERMSprop/gru_cell/kernel/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE%RMSprop/gru_cell/recurrent_kernel/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUERMSprop/gru_cell/bias/rmsEvariables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUERMSprop/gru_cell_1/kernel/rmsEvariables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'RMSprop/gru_cell_1/recurrent_kernel/rmsEvariables/16/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUERMSprop/gru_cell_1/bias/rmsEvariables/17/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUERMSprop/dense_3/kernel/rmsEvariables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUERMSprop/dense_3/bias/rmsEvariables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUERMSprop/dense_4/kernel/rmsEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUERMSprop/dense_4/bias/rmsEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUERMSprop/dense/kernel/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUERMSprop/dense/bias/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUERMSprop/dense_1/kernel/momentumIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUERMSprop/dense_1/bias/momentumIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUERMSprop/dense_2/kernel/momentumIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUERMSprop/dense_2/bias/momentumIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*RMSprop/batch_normalization/gamma/momentumIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)RMSprop/batch_normalization/beta/momentumIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%RMSprop/embedding/embeddings/momentumJvariables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'RMSprop/embedding_1/embeddings/momentumJvariables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE RMSprop/gru_cell/kernel/momentumJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*RMSprop/gru_cell/recurrent_kernel/momentumJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUERMSprop/gru_cell/bias/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"RMSprop/gru_cell_1/kernel/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,RMSprop/gru_cell_1/recurrent_kernel/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE RMSprop/gru_cell_1/bias/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUERMSprop/dense_3/kernel/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUERMSprop/dense_3/bias/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUERMSprop/dense_4/kernel/momentumJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUERMSprop/dense_4/bias/momentumJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
z
serving_default_input_2Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
z
serving_default_input_3Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
z
serving_default_input_4Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_input_5Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿd
¨
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3serving_default_input_4serving_default_input_5dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasbatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization/betabatch_normalization/gammaembedding/embeddingsembedding_1/embeddingsgru_cell/biasgru_cell/kernelgru_cell/recurrent_kernelgru_cell_1/biasgru_cell_1/kernelgru_cell_1/recurrent_kerneldense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_337801899
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Í
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp(embedding/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp#gru_cell/kernel/Read/ReadVariableOp-gru_cell/recurrent_kernel/Read/ReadVariableOp!gru_cell/bias/Read/ReadVariableOp%gru_cell_1/kernel/Read/ReadVariableOp/gru_cell_1/recurrent_kernel/Read/ReadVariableOp#gru_cell_1/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOp.RMSprop/dense_2/kernel/rms/Read/ReadVariableOp,RMSprop/dense_2/bias/rms/Read/ReadVariableOp9RMSprop/batch_normalization/gamma/rms/Read/ReadVariableOp8RMSprop/batch_normalization/beta/rms/Read/ReadVariableOp4RMSprop/embedding/embeddings/rms/Read/ReadVariableOp6RMSprop/embedding_1/embeddings/rms/Read/ReadVariableOp/RMSprop/gru_cell/kernel/rms/Read/ReadVariableOp9RMSprop/gru_cell/recurrent_kernel/rms/Read/ReadVariableOp-RMSprop/gru_cell/bias/rms/Read/ReadVariableOp1RMSprop/gru_cell_1/kernel/rms/Read/ReadVariableOp;RMSprop/gru_cell_1/recurrent_kernel/rms/Read/ReadVariableOp/RMSprop/gru_cell_1/bias/rms/Read/ReadVariableOp.RMSprop/dense_3/kernel/rms/Read/ReadVariableOp,RMSprop/dense_3/bias/rms/Read/ReadVariableOp.RMSprop/dense_4/kernel/rms/Read/ReadVariableOp,RMSprop/dense_4/bias/rms/Read/ReadVariableOp1RMSprop/dense/kernel/momentum/Read/ReadVariableOp/RMSprop/dense/bias/momentum/Read/ReadVariableOp3RMSprop/dense_1/kernel/momentum/Read/ReadVariableOp1RMSprop/dense_1/bias/momentum/Read/ReadVariableOp3RMSprop/dense_2/kernel/momentum/Read/ReadVariableOp1RMSprop/dense_2/bias/momentum/Read/ReadVariableOp>RMSprop/batch_normalization/gamma/momentum/Read/ReadVariableOp=RMSprop/batch_normalization/beta/momentum/Read/ReadVariableOp9RMSprop/embedding/embeddings/momentum/Read/ReadVariableOp;RMSprop/embedding_1/embeddings/momentum/Read/ReadVariableOp4RMSprop/gru_cell/kernel/momentum/Read/ReadVariableOp>RMSprop/gru_cell/recurrent_kernel/momentum/Read/ReadVariableOp2RMSprop/gru_cell/bias/momentum/Read/ReadVariableOp6RMSprop/gru_cell_1/kernel/momentum/Read/ReadVariableOp@RMSprop/gru_cell_1/recurrent_kernel/momentum/Read/ReadVariableOp4RMSprop/gru_cell_1/bias/momentum/Read/ReadVariableOp3RMSprop/dense_3/kernel/momentum/Read/ReadVariableOp1RMSprop/dense_3/bias/momentum/Read/ReadVariableOp3RMSprop/dense_4/kernel/momentum/Read/ReadVariableOp1RMSprop/dense_4/bias/momentum/Read/ReadVariableOpConst*P
TinI
G2E	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_save_337803245

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceembedding/embeddingsembedding_1/embeddingsgru_cell/kernelgru_cell/recurrent_kernelgru_cell/biasgru_cell_1/kernelgru_cell_1/recurrent_kernelgru_cell_1/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhoRMSprop/dense/kernel/rmsRMSprop/dense/bias/rmsRMSprop/dense_1/kernel/rmsRMSprop/dense_1/bias/rmsRMSprop/dense_2/kernel/rmsRMSprop/dense_2/bias/rms%RMSprop/batch_normalization/gamma/rms$RMSprop/batch_normalization/beta/rms RMSprop/embedding/embeddings/rms"RMSprop/embedding_1/embeddings/rmsRMSprop/gru_cell/kernel/rms%RMSprop/gru_cell/recurrent_kernel/rmsRMSprop/gru_cell/bias/rmsRMSprop/gru_cell_1/kernel/rms'RMSprop/gru_cell_1/recurrent_kernel/rmsRMSprop/gru_cell_1/bias/rmsRMSprop/dense_3/kernel/rmsRMSprop/dense_3/bias/rmsRMSprop/dense_4/kernel/rmsRMSprop/dense_4/bias/rmsRMSprop/dense/kernel/momentumRMSprop/dense/bias/momentumRMSprop/dense_1/kernel/momentumRMSprop/dense_1/bias/momentumRMSprop/dense_2/kernel/momentumRMSprop/dense_2/bias/momentum*RMSprop/batch_normalization/gamma/momentum)RMSprop/batch_normalization/beta/momentum%RMSprop/embedding/embeddings/momentum'RMSprop/embedding_1/embeddings/momentum RMSprop/gru_cell/kernel/momentum*RMSprop/gru_cell/recurrent_kernel/momentumRMSprop/gru_cell/bias/momentum"RMSprop/gru_cell_1/kernel/momentum,RMSprop/gru_cell_1/recurrent_kernel/momentum RMSprop/gru_cell_1/bias/momentumRMSprop/dense_3/kernel/momentumRMSprop/dense_3/bias/momentumRMSprop/dense_4/kernel/momentumRMSprop/dense_4/bias/momentum*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__traced_restore_337803456

Ú
G__inference_gru_cell_layer_call_and_return_conditional_losses_337800181

inputs

states*
readvariableop_resource:	¬2
matmul_readvariableop_resource:
¬3
 matmul_1_readvariableop_resource:	d¬
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
¼
Á
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800052
dense_3_input#
dense_3_337800041:d@
dense_3_337800043:@#
dense_4_337800046:@
dense_4_337800048:
identity¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCallü
dense_3/StatefulPartitionedCallStatefulPartitionedCalldense_3_inputdense_3_337800041dense_3_337800043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_337799916
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_337800046dense_4_337800048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_337799933w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'
_user_specified_namedense_3_input

´
I__inference_sequential_layer_call_and_return_conditional_losses_337799600

inputs!
dense_337799589:@
dense_337799591:@$
dense_1_337799594:	@ 
dense_1_337799596:	
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallí
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_337799589dense_337799591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_337799516
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_337799594dense_1_337799596*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_337799533x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Á
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800038
dense_3_input#
dense_3_337800027:d@
dense_3_337800029:@#
dense_4_337800032:@
dense_4_337800034:
identity¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCallü
dense_3/StatefulPartitionedCallStatefulPartitionedCalldense_3_inputdense_3_337800027dense_3_337800029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_337799916
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_337800032dense_4_337800034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_337799933w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'
_user_specified_namedense_3_input
	

0__inference_sequential_1_layer_call_fn_337799783
dense_2_input
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799768p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_2_input
§
º
K__inference_sequential_2_layer_call_and_return_conditional_losses_337799940

inputs#
dense_3_337799917:d@
dense_3_337799919:@#
dense_4_337799934:@
dense_4_337799936:
identity¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCallõ
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_337799917dense_3_337799919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_337799916
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_337799934dense_4_337799936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_337799933w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
"
Ö
K__inference_sequential_1_layer_call_and_return_conditional_losses_337802468

inputs9
&dense_2_matmul_readvariableop_resource:	6
'dense_2_biasadd_readvariableop_resource:	?
0batch_normalization_cast_readvariableop_resource:	A
2batch_normalization_cast_1_readvariableop_resource:	A
2batch_normalization_cast_2_readvariableop_resource:	A
2batch_normalization_cast_3_readvariableop_resource:	
identity¢'batch_normalization/Cast/ReadVariableOp¢)batch_normalization/Cast_1/ReadVariableOp¢)batch_normalization/Cast_2/ReadVariableOp¢)batch_normalization/Cast_3/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0z
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:*
dtype0
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:±
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ª
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
: 
#batch_normalization/batchnorm/mul_1Muldense_2/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:ª
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¯
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity'batch_normalization/batchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_4_layer_call_and_return_conditional_losses_337799933

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¥

ù
F__inference_dense_2_layer_call_and_return_conditional_losses_337799752

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
º
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799768

inputs$
dense_2_337799753:	 
dense_2_337799755:	,
batch_normalization_337799758:	,
batch_normalization_337799760:	,
batch_normalization_337799762:	,
batch_normalization_337799764:	
identity¢+batch_normalization/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallö
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_337799753dense_2_337799755*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_337799752
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_337799758batch_normalization_337799760batch_normalization_337799762batch_normalization_337799764*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337799676
IdentityIdentity4batch_normalization/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
E
§

C__inference_rial_layer_call_and_return_conditional_losses_337800880
input_1
input_2
input_3
input_4
input_5&
sequential_337800808:@"
sequential_337800810:@'
sequential_337800812:	@#
sequential_337800814:	)
sequential_1_337800818:	%
sequential_1_337800820:	%
sequential_1_337800822:	%
sequential_1_337800824:	%
sequential_1_337800826:	%
sequential_1_337800828:	&
embedding_337800832:	(
embedding_1_337800836:	%
gru_cell_337800848:	¬&
gru_cell_337800850:
¬%
gru_cell_337800852:	d¬'
gru_cell_1_337800860:	¬'
gru_cell_1_337800862:	d¬'
gru_cell_1_337800864:	d¬(
sequential_2_337800868:d@$
sequential_2_337800870:@(
sequential_2_337800872:@$
sequential_2_337800874:
identity

identity_1

identity_2¢!embedding/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢ gru_cell/StatefulPartitionedCall¢"gru_cell_1/StatefulPartitionedCall¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_2/StatefulPartitionedCallc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinput_5transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd³
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_337800808sequential_337800810sequential_337800812sequential_337800814*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_337799600V
CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallCast:y:0sequential_1_337800818sequential_1_337800820sequential_1_337800822sequential_1_337800824sequential_1_337800826sequential_1_337800828*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799830X
Cast_1Castinput_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
!embedding/StatefulPartitionedCallStatefulPartitionedCall
Cast_1:y:0embedding_337800832*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_layer_call_and_return_conditional_losses_337800103X
Cast_2Castinput_4*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall
Cast_2:y:0embedding_1_337800836*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_337800118^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape*embedding/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape,embedding_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
add/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0Reshape:output:0-sequential_1/StatefulPartitionedCall:output:0Reshape_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_layer_call_and_return_conditional_losses_337800136]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskÒ
 gru_cell/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0strided_slice:output:0gru_cell_337800848gru_cell_337800850gru_cell_337800852*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_layer_call_and_return_conditional_losses_337800430_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskë
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall)gru_cell/StatefulPartitionedCall:output:0strided_slice_1:output:0gru_cell_1_337800860gru_cell_1_337800862gru_cell_1_337800864*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337800362â
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall+gru_cell_1/StatefulPartitionedCall:output:0sequential_2_337800868sequential_2_337800870sequential_2_337800872sequential_2_337800874*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800000|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz

Identity_1Identity)gru_cell/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|

Identity_2Identity+gru_cell_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdË
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall!^gru_cell/StatefulPartitionedCall#^gru_cell_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_5
Ð$
×
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337799723

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	+
cast_readvariableop_resource:	-
cast_1_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
Æ
I__inference_sequential_layer_call_and_return_conditional_losses_337802407

inputs6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@9
&dense_1_matmul_readvariableop_resource:	@6
'dense_1_biasadd_readvariableop_resource:	
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
Á
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799880
dense_2_input$
dense_2_337799865:	 
dense_2_337799867:	,
batch_normalization_337799870:	,
batch_normalization_337799872:	,
batch_normalization_337799874:	,
batch_normalization_337799876:	
identity¢+batch_normalization/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallý
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_337799865dense_2_337799867*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_337799752
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_337799870batch_normalization_337799872batch_normalization_337799874batch_normalization_337799876*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337799676
IdentityIdentity4batch_normalization/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_2_input
Ã²

C__inference_rial_layer_call_and_return_conditional_losses_337802172
input_0
input_1
input_2
input_3
input_4A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@D
1sequential_dense_1_matmul_readvariableop_resource:	@A
2sequential_dense_1_biasadd_readvariableop_resource:	F
3sequential_1_dense_2_matmul_readvariableop_resource:	C
4sequential_1_dense_2_biasadd_readvariableop_resource:	L
=sequential_1_batch_normalization_cast_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_1_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_2_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_3_readvariableop_resource:	7
$embedding_embedding_lookup_337802062:	9
&embedding_1_embedding_lookup_337802069:	3
 gru_cell_readvariableop_resource:	¬;
'gru_cell_matmul_readvariableop_resource:
¬<
)gru_cell_matmul_1_readvariableop_resource:	d¬5
"gru_cell_1_readvariableop_resource:	¬<
)gru_cell_1_matmul_readvariableop_resource:	d¬>
+gru_cell_1_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_3_matmul_readvariableop_resource:d@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢embedding/embedding_lookup¢embedding_1/embedding_lookup¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢4sequential_1/batch_normalization/Cast/ReadVariableOp¢6sequential_1/batch_normalization/Cast_1/ReadVariableOp¢6sequential_1/batch_normalization/Cast_2/ReadVariableOp¢6sequential_1/batch_normalization/Cast_3/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOpc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinput_4transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
sequential/dense/MatMulMatMulinput_0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0­
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential_1/dense_2/MatMulMatMulCast:y:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¶
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
4sequential_1/batch_normalization/Cast/ReadVariableOpReadVariableOp=sequential_1_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_2/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_3/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:*
dtype0u
0sequential_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ø
.sequential_1/batch_normalization/batchnorm/addAddV2>sequential_1/batch_normalization/Cast_1/ReadVariableOp:value:09sequential_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0sequential_1/batch_normalization/batchnorm/RsqrtRsqrt2sequential_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/mulMul4sequential_1/batch_normalization/batchnorm/Rsqrt:y:0>sequential_1/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ç
0sequential_1/batch_normalization/batchnorm/mul_1Mul'sequential_1/dense_2/Relu:activations:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
0sequential_1/batch_normalization/batchnorm/mul_2Mul<sequential_1/batch_normalization/Cast/ReadVariableOp:value:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/subSub>sequential_1/batch_normalization/Cast_2/ReadVariableOp:value:04sequential_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ö
0sequential_1/batch_normalization/batchnorm/add_1AddV24sequential_1/batch_normalization/batchnorm/mul_1:z:02sequential_1/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Cast_1Castinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
embedding/CastCast
Cast_1:y:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
embedding/embedding_lookupResourceGather$embedding_embedding_lookup_337802062embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding/embedding_lookup/337802062*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ä
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding/embedding_lookup/337802062*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Cast_2Castinput_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
embedding_1/CastCast
Cast_2:y:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_337802069embedding_1/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/337802069*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ê
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/337802069*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape0embedding_1/embedding_lookup/Identity_1:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
add/addAddV2%sequential/dense_1/Relu:activations:0Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
	add/add_1AddV2add/add:z:04sequential_1/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
	add/add_2AddV2add/add_1:z:0Reshape_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	¬*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0
gru_cell/MatMulMatMuladd/add_2:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell/MatMul_1MatMulstrided_slice:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMulMatMulgru_cell/add_3:z:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0¡
sequential_2/dense_3/MatMulMatMulgru_cell_1/add_3:z:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0´
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_2/dense_4/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc

Identity_1Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde

Identity_2Identitygru_cell_1/add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdá
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp5^sequential_1/batch_normalization/Cast/ReadVariableOp7^sequential_1/batch_normalization/Cast_1/ReadVariableOp7^sequential_1/batch_normalization/Cast_2/ReadVariableOp7^sequential_1/batch_normalization/Cast_3/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2l
4sequential_1/batch_normalization/Cast/ReadVariableOp4sequential_1/batch_normalization/Cast/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_1/ReadVariableOp6sequential_1/batch_normalization/Cast_1/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_2/ReadVariableOp6sequential_1/batch_normalization/Cast_2/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_3/ReadVariableOp6sequential_1/batch_normalization/Cast_3/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/3:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input/4
ã;
Ì
K__inference_sequential_1_layer_call_and_return_conditional_losses_337802509

inputs9
&dense_2_matmul_readvariableop_resource:	6
'dense_2_biasadd_readvariableop_resource:	J
;batch_normalization_assignmovingavg_readvariableop_resource:	L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	?
0batch_normalization_cast_readvariableop_resource:	A
2batch_normalization_cast_1_readvariableop_resource:	
identity¢#batch_normalization/AssignMovingAvg¢2batch_normalization/AssignMovingAvg/ReadVariableOp¢%batch_normalization/AssignMovingAvg_1¢4batch_normalization/AssignMovingAvg_1/ReadVariableOp¢'batch_normalization/Cast/ReadVariableOp¢)batch_normalization/Cast_1/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0z
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¼
 batch_normalization/moments/meanMeandense_2/Relu:activations:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	Ä
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense_2/Relu:activations:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Û
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<«
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0¾
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:µ
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ü
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¯
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:»
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:®
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ª
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
: 
#batch_normalization/batchnorm/mul_1Muldense_2/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:¨
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¯
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity'batch_normalization/batchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
Î
K__inference_sequential_2_layer_call_and_return_conditional_losses_337802817

inputs8
&dense_3_matmul_readvariableop_resource:d@5
'dense_3_biasadd_readvariableop_resource:@8
&dense_4_matmul_readvariableop_resource:@5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0y
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_4/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
øÐ
¨
C__inference_rial_layer_call_and_return_conditional_losses_337802345
input_0
input_1
input_2
input_3
input_4A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@D
1sequential_dense_1_matmul_readvariableop_resource:	@A
2sequential_dense_1_biasadd_readvariableop_resource:	F
3sequential_1_dense_2_matmul_readvariableop_resource:	C
4sequential_1_dense_2_biasadd_readvariableop_resource:	W
Hsequential_1_batch_normalization_assignmovingavg_readvariableop_resource:	Y
Jsequential_1_batch_normalization_assignmovingavg_1_readvariableop_resource:	L
=sequential_1_batch_normalization_cast_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_1_readvariableop_resource:	7
$embedding_embedding_lookup_337802235:	9
&embedding_1_embedding_lookup_337802242:	3
 gru_cell_readvariableop_resource:	¬;
'gru_cell_matmul_readvariableop_resource:
¬<
)gru_cell_matmul_1_readvariableop_resource:	d¬5
"gru_cell_1_readvariableop_resource:	¬<
)gru_cell_1_matmul_readvariableop_resource:	d¬>
+gru_cell_1_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_3_matmul_readvariableop_resource:d@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢embedding/embedding_lookup¢embedding_1/embedding_lookup¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢0sequential_1/batch_normalization/AssignMovingAvg¢?sequential_1/batch_normalization/AssignMovingAvg/ReadVariableOp¢2sequential_1/batch_normalization/AssignMovingAvg_1¢Asequential_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp¢4sequential_1/batch_normalization/Cast/ReadVariableOp¢6sequential_1/batch_normalization/Cast_1/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOpc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinput_4transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
sequential/dense/MatMulMatMulinput_0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0­
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential_1/dense_2/MatMulMatMulCast:y:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¶
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?sequential_1/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
-sequential_1/batch_normalization/moments/meanMean'sequential_1/dense_2/Relu:activations:0Hsequential_1/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(§
5sequential_1/batch_normalization/moments/StopGradientStopGradient6sequential_1/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	ë
:sequential_1/batch_normalization/moments/SquaredDifferenceSquaredDifference'sequential_1/dense_2/Relu:activations:0>sequential_1/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Csequential_1/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
1sequential_1/batch_normalization/moments/varianceMean>sequential_1/batch_normalization/moments/SquaredDifference:z:0Lsequential_1/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(°
0sequential_1/batch_normalization/moments/SqueezeSqueeze6sequential_1/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 ¶
2sequential_1/batch_normalization/moments/Squeeze_1Squeeze:sequential_1/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 {
6sequential_1/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Å
?sequential_1/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpHsequential_1_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0å
4sequential_1/batch_normalization/AssignMovingAvg/subSubGsequential_1/batch_normalization/AssignMovingAvg/ReadVariableOp:value:09sequential_1/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü
4sequential_1/batch_normalization/AssignMovingAvg/mulMul8sequential_1/batch_normalization/AssignMovingAvg/sub:z:0?sequential_1/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:°
0sequential_1/batch_normalization/AssignMovingAvgAssignSubVariableOpHsequential_1_batch_normalization_assignmovingavg_readvariableop_resource8sequential_1/batch_normalization/AssignMovingAvg/mul:z:0@^sequential_1/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0}
8sequential_1/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<É
Asequential_1/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpJsequential_1_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0ë
6sequential_1/batch_normalization/AssignMovingAvg_1/subSubIsequential_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0;sequential_1/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:â
6sequential_1/batch_normalization/AssignMovingAvg_1/mulMul:sequential_1/batch_normalization/AssignMovingAvg_1/sub:z:0Asequential_1/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:¸
2sequential_1/batch_normalization/AssignMovingAvg_1AssignSubVariableOpJsequential_1_batch_normalization_assignmovingavg_1_readvariableop_resource:sequential_1/batch_normalization/AssignMovingAvg_1/mul:z:0B^sequential_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0¯
4sequential_1/batch_normalization/Cast/ReadVariableOpReadVariableOp=sequential_1_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0u
0sequential_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Õ
.sequential_1/batch_normalization/batchnorm/addAddV2;sequential_1/batch_normalization/moments/Squeeze_1:output:09sequential_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0sequential_1/batch_normalization/batchnorm/RsqrtRsqrt2sequential_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/mulMul4sequential_1/batch_normalization/batchnorm/Rsqrt:y:0>sequential_1/batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ç
0sequential_1/batch_normalization/batchnorm/mul_1Mul'sequential_1/dense_2/Relu:activations:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
0sequential_1/batch_normalization/batchnorm/mul_2Mul9sequential_1/batch_normalization/moments/Squeeze:output:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ï
.sequential_1/batch_normalization/batchnorm/subSub<sequential_1/batch_normalization/Cast/ReadVariableOp:value:04sequential_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ö
0sequential_1/batch_normalization/batchnorm/add_1AddV24sequential_1/batch_normalization/batchnorm/mul_1:z:02sequential_1/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Cast_1Castinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
embedding/CastCast
Cast_1:y:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
embedding/embedding_lookupResourceGather$embedding_embedding_lookup_337802235embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding/embedding_lookup/337802235*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ä
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding/embedding_lookup/337802235*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Cast_2Castinput_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
embedding_1/CastCast
Cast_2:y:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_337802242embedding_1/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/337802242*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ê
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/337802242*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape0embedding_1/embedding_lookup/Identity_1:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
add/addAddV2%sequential/dense_1/Relu:activations:0Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
	add/add_1AddV2add/add:z:04sequential_1/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
	add/add_2AddV2add/add_1:z:0Reshape_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	¬*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0
gru_cell/MatMulMatMuladd/add_2:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell/MatMul_1MatMulstrided_slice:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMulMatMulgru_cell/add_3:z:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0¡
sequential_2/dense_3/MatMulMatMulgru_cell_1/add_3:z:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0´
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_2/dense_4/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc

Identity_1Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde

Identity_2Identitygru_cell_1/add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÝ
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp1^sequential_1/batch_normalization/AssignMovingAvg@^sequential_1/batch_normalization/AssignMovingAvg/ReadVariableOp3^sequential_1/batch_normalization/AssignMovingAvg_1B^sequential_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp5^sequential_1/batch_normalization/Cast/ReadVariableOp7^sequential_1/batch_normalization/Cast_1/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2d
0sequential_1/batch_normalization/AssignMovingAvg0sequential_1/batch_normalization/AssignMovingAvg2
?sequential_1/batch_normalization/AssignMovingAvg/ReadVariableOp?sequential_1/batch_normalization/AssignMovingAvg/ReadVariableOp2h
2sequential_1/batch_normalization/AssignMovingAvg_12sequential_1/batch_normalization/AssignMovingAvg_12
Asequential_1/batch_normalization/AssignMovingAvg_1/ReadVariableOpAsequential_1/batch_normalization/AssignMovingAvg_1/ReadVariableOp2l
4sequential_1/batch_normalization/Cast/ReadVariableOp4sequential_1/batch_normalization/Cast/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_1/ReadVariableOp6sequential_1/batch_normalization/Cast_1/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/3:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input/4
Ê

+__inference_dense_2_layer_call_fn_337802884

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_337799752p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ	
©
H__inference_embedding_layer_call_and_return_conditional_losses_337802526

inputs-
embedding_lookup_337802520:	
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
embedding_lookupResourceGatherembedding_lookup_337802520Cast:y:0*
Tindices0*-
_class#
!loc:@embedding_lookup/337802520*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¦
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/337802520*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Û
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337802773

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
²

Û
.__inference_gru_cell_1_layer_call_fn_337802695

inputs

states
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatesunknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337800362o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
â

B__inference_add_layer_call_and_return_conditional_losses_337802561
inputs_0
inputs_1
inputs_2
inputs_3
identityS
addAddV2inputs_0inputs_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2add:z:0inputs_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_2AddV2	add_1:z:0inputs_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3

©
'__inference_signature_wrapper_337801899
input_1
input_2
input_3
input_4
input_5
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	¬

unknown_12:
¬

unknown_13:	d¬

unknown_14:	¬

unknown_15:	d¬

unknown_16:	d¬

unknown_17:d@

unknown_18:@

unknown_19:@

unknown_20:
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*&
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_337799498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_5
©
ª
(__inference_rial_layer_call_fn_337801956
input_0
input_1
input_2
input_3
input_4
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	¬

unknown_12:
¬

unknown_13:	d¬

unknown_14:	¬

unknown_15:	d¬

unknown_16:	d¬

unknown_17:d@

unknown_18:@

unknown_19:@

unknown_20:
identity

identity_1

identity_2¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*&
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_rial_layer_call_and_return_conditional_losses_337800254o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/3:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input/4
Â

)__inference_dense_layer_call_fn_337802844

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_337799516o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
Ó
.__inference_sequential_layer_call_fn_337802371

inputs
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_337799600p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Û
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337800233

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
±

Ú
,__inference_gru_cell_layer_call_fn_337802575

inputs

states
unknown:	¬
	unknown_0:
¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatesunknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_layer_call_and_return_conditional_losses_337800181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
©
ª
(__inference_rial_layer_call_fn_337800305
input_1
input_2
input_3
input_4
input_5
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	¬

unknown_12:
¬

unknown_13:	d¬

unknown_14:	¬

unknown_15:	d¬

unknown_16:	d¬

unknown_17:d@

unknown_18:@

unknown_19:@

unknown_20:
identity

identity_1

identity_2¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*&
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_rial_layer_call_and_return_conditional_losses_337800254o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_5
±

R__inference_batch_normalization_layer_call_and_return_conditional_losses_337802941

inputs+
cast_readvariableop_resource:	-
cast_1_readvariableop_resource:	-
cast_2_readvariableop_resource:	-
cast_3_readvariableop_resource:	
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð$
×
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337802975

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	+
cast_readvariableop_resource:	-
cast_1_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


õ
D__inference_dense_layer_call_and_return_conditional_losses_337799516

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ú
G__inference_gru_cell_layer_call_and_return_conditional_losses_337802628

inputs

states*
readvariableop_resource:	¬2
matmul_readvariableop_resource:
¬3
 matmul_1_readvariableop_resource:	d¬
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
Þ«
ï
__inference_call_337801522
input_0
input_1	
input_2
input_3
input_4A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@D
1sequential_dense_1_matmul_readvariableop_resource:	@A
2sequential_dense_1_biasadd_readvariableop_resource:	F
3sequential_1_dense_2_matmul_readvariableop_resource:	C
4sequential_1_dense_2_biasadd_readvariableop_resource:	L
=sequential_1_batch_normalization_cast_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_1_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_2_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_3_readvariableop_resource:	7
$embedding_embedding_lookup_337801412:	9
&embedding_1_embedding_lookup_337801419:	3
 gru_cell_readvariableop_resource:	¬;
'gru_cell_matmul_readvariableop_resource:
¬<
)gru_cell_matmul_1_readvariableop_resource:	d¬5
"gru_cell_1_readvariableop_resource:	¬<
)gru_cell_1_matmul_readvariableop_resource:	d¬>
+gru_cell_1_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_3_matmul_readvariableop_resource:d@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢embedding/embedding_lookup¢embedding_1/embedding_lookup¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢4sequential_1/batch_normalization/Cast/ReadVariableOp¢6sequential_1/batch_normalization/Cast_1/ReadVariableOp¢6sequential_1/batch_normalization/Cast_2/ReadVariableOp¢6sequential_1/batch_normalization/Cast_3/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOpc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          e
	transpose	Transposeinput_4transpose/perm:output:0*
T0*"
_output_shapes
:d
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
sequential/dense/MatMulMatMulinput_0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@i
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes

:@
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¤
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	n
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	M
CastCastinput_2*

DstT0*

SrcT0*
_output_shapes

:
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential_1/dense_2/MatMulMatMulCast:y:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	r
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:	¯
4sequential_1/batch_normalization/Cast/ReadVariableOpReadVariableOp=sequential_1_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_2/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_3/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:*
dtype0u
0sequential_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ø
.sequential_1/batch_normalization/batchnorm/addAddV2>sequential_1/batch_normalization/Cast_1/ReadVariableOp:value:09sequential_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0sequential_1/batch_normalization/batchnorm/RsqrtRsqrt2sequential_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/mulMul4sequential_1/batch_normalization/batchnorm/Rsqrt:y:0>sequential_1/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:¾
0sequential_1/batch_normalization/batchnorm/mul_1Mul'sequential_1/dense_2/Relu:activations:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:	Ï
0sequential_1/batch_normalization/batchnorm/mul_2Mul<sequential_1/batch_normalization/Cast/ReadVariableOp:value:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/subSub>sequential_1/batch_normalization/Cast_2/ReadVariableOp:value:04sequential_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Í
0sequential_1/batch_normalization/batchnorm/add_1AddV24sequential_1/batch_normalization/batchnorm/mul_1:z:02sequential_1/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes
:	O
Cast_1Castinput_1*

DstT0*

SrcT0	*
_output_shapes

:Z
embedding/CastCast
Cast_1:y:0*

DstT0*

SrcT0*
_output_shapes

:á
embedding/embedding_lookupResourceGather$embedding_embedding_lookup_337801412embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding/embedding_lookup/337801412*#
_output_shapes
:*
dtype0»
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding/embedding_lookup/337801412*#
_output_shapes
:
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
:O
Cast_2Castinput_3*

DstT0*

SrcT0*
_output_shapes

:\
embedding_1/CastCast
Cast_2:y:0*

DstT0*

SrcT0*
_output_shapes

:é
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_337801419embedding_1/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801419*#
_output_shapes
:*
dtype0Á
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801419*#
_output_shapes
:
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
:^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0Reshape/shape:output:0*
T0*
_output_shapes
:	`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape0embedding_1/embedding_lookup/Identity_1:output:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	s
add/addAddV2%sequential/dense_1/Relu:activations:0Reshape:output:0*
T0*
_output_shapes
:	
	add/add_1AddV2add/add:z:04sequential_1/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:	_
	add/add_2AddV2add/add_1:z:0Reshape_1:output:0*
T0*
_output_shapes
:	]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	¬*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0z
gru_cell/MatMulMatMuladd/add_2:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬{
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	¬c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:d:d:d*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell/MatMul_1MatMulstrided_slice:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	¬c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÏ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
:d:d:d*
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:dV
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:dt
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:dZ
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:do
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes

:dk
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes

:dR
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:dl
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0strided_slice:output:0*
T0*
_output_shapes

:dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:dc
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:dh
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:d_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMulMatMulgru_cell/add_3:z:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*
_output_shapes
:	¬e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*2
_output_shapes 
:d:d:d*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*
_output_shapes
:	¬e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ×
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
:d:d:d*
	num_splitx
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*
_output_shapes

:dZ
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes

:dz
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*
_output_shapes

:d^
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes

:du
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*
_output_shapes

:dq
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*
_output_shapes

:dV
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes

:dr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0strided_slice_1:output:0*
T0*
_output_shapes

:dU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?q
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes

:di
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes

:dn
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes

:d
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0
sequential_2/dense_3/MatMulMatMulgru_cell_1/add_3:z:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¬
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@q
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*
_output_shapes

:@
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0«
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:q
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*
_output_shapes

:m
IdentityIdentity'sequential_2/dense_4/Relu:activations:0^NoOp*
T0*
_output_shapes

:Z

Identity_1Identitygru_cell/add_3:z:0^NoOp*
T0*
_output_shapes

:d\

Identity_2Identitygru_cell_1/add_3:z:0^NoOp*
T0*
_output_shapes

:dá
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp5^sequential_1/batch_normalization/Cast/ReadVariableOp7^sequential_1/batch_normalization/Cast_1/ReadVariableOp7^sequential_1/batch_normalization/Cast_2/ReadVariableOp7^sequential_1/batch_normalization/Cast_3/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:::::d: : : : : : : : : : : : : : : : : : : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2l
4sequential_1/batch_normalization/Cast/ReadVariableOp4sequential_1/batch_normalization/Cast/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_1/ReadVariableOp6sequential_1/batch_normalization/Cast_1/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_2/ReadVariableOp6sequential_1/batch_normalization/Cast_2/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_3/ReadVariableOp6sequential_1/batch_normalization/Cast_3/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:G C

_output_shapes

:
!
_user_specified_name	input/0:GC

_output_shapes

:
!
_user_specified_name	input/1:GC

_output_shapes

:
!
_user_specified_name	input/2:GC

_output_shapes

:
!
_user_specified_name	input/3:KG
"
_output_shapes
:d
!
_user_specified_name	input/4

Ú
G__inference_gru_cell_layer_call_and_return_conditional_losses_337802667

inputs

states*
readvariableop_resource:	¬2
matmul_readvariableop_resource:
¬3
 matmul_1_readvariableop_resource:	d¬
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
¹
Ú
0__inference_sequential_2_layer_call_fn_337800024
dense_3_input
unknown:d@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_3_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800000o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'
_user_specified_namedense_3_input
¤
Ó
0__inference_sequential_2_layer_call_fn_337802786

inputs
unknown:d@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_337799940o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¦
o
'__inference_add_layer_call_fn_337802551
inputs_0
inputs_1
inputs_2
inputs_3
identityÑ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_layer_call_and_return_conditional_losses_337800136a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/3


õ
D__inference_dense_layer_call_and_return_conditional_losses_337802855

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ã-
%__inference__traced_restore_337803456
file_prefix/
assignvariableop_dense_kernel:@+
assignvariableop_1_dense_bias:@4
!assignvariableop_2_dense_1_kernel:	@.
assignvariableop_3_dense_1_bias:	4
!assignvariableop_4_dense_2_kernel:	.
assignvariableop_5_dense_2_bias:	;
,assignvariableop_6_batch_normalization_gamma:	:
+assignvariableop_7_batch_normalization_beta:	A
2assignvariableop_8_batch_normalization_moving_mean:	E
6assignvariableop_9_batch_normalization_moving_variance:	;
(assignvariableop_10_embedding_embeddings:	=
*assignvariableop_11_embedding_1_embeddings:	7
#assignvariableop_12_gru_cell_kernel:
¬@
-assignvariableop_13_gru_cell_recurrent_kernel:	d¬4
!assignvariableop_14_gru_cell_bias:	¬8
%assignvariableop_15_gru_cell_1_kernel:	d¬B
/assignvariableop_16_gru_cell_1_recurrent_kernel:	d¬6
#assignvariableop_17_gru_cell_1_bias:	¬4
"assignvariableop_18_dense_3_kernel:d@.
 assignvariableop_19_dense_3_bias:@4
"assignvariableop_20_dense_4_kernel:@.
 assignvariableop_21_dense_4_bias:*
 assignvariableop_22_rmsprop_iter:	 +
!assignvariableop_23_rmsprop_decay: 3
)assignvariableop_24_rmsprop_learning_rate: .
$assignvariableop_25_rmsprop_momentum: )
assignvariableop_26_rmsprop_rho: >
,assignvariableop_27_rmsprop_dense_kernel_rms:@8
*assignvariableop_28_rmsprop_dense_bias_rms:@A
.assignvariableop_29_rmsprop_dense_1_kernel_rms:	@;
,assignvariableop_30_rmsprop_dense_1_bias_rms:	A
.assignvariableop_31_rmsprop_dense_2_kernel_rms:	;
,assignvariableop_32_rmsprop_dense_2_bias_rms:	H
9assignvariableop_33_rmsprop_batch_normalization_gamma_rms:	G
8assignvariableop_34_rmsprop_batch_normalization_beta_rms:	G
4assignvariableop_35_rmsprop_embedding_embeddings_rms:	I
6assignvariableop_36_rmsprop_embedding_1_embeddings_rms:	C
/assignvariableop_37_rmsprop_gru_cell_kernel_rms:
¬L
9assignvariableop_38_rmsprop_gru_cell_recurrent_kernel_rms:	d¬@
-assignvariableop_39_rmsprop_gru_cell_bias_rms:	¬D
1assignvariableop_40_rmsprop_gru_cell_1_kernel_rms:	d¬N
;assignvariableop_41_rmsprop_gru_cell_1_recurrent_kernel_rms:	d¬B
/assignvariableop_42_rmsprop_gru_cell_1_bias_rms:	¬@
.assignvariableop_43_rmsprop_dense_3_kernel_rms:d@:
,assignvariableop_44_rmsprop_dense_3_bias_rms:@@
.assignvariableop_45_rmsprop_dense_4_kernel_rms:@:
,assignvariableop_46_rmsprop_dense_4_bias_rms:C
1assignvariableop_47_rmsprop_dense_kernel_momentum:@=
/assignvariableop_48_rmsprop_dense_bias_momentum:@F
3assignvariableop_49_rmsprop_dense_1_kernel_momentum:	@@
1assignvariableop_50_rmsprop_dense_1_bias_momentum:	F
3assignvariableop_51_rmsprop_dense_2_kernel_momentum:	@
1assignvariableop_52_rmsprop_dense_2_bias_momentum:	M
>assignvariableop_53_rmsprop_batch_normalization_gamma_momentum:	L
=assignvariableop_54_rmsprop_batch_normalization_beta_momentum:	L
9assignvariableop_55_rmsprop_embedding_embeddings_momentum:	N
;assignvariableop_56_rmsprop_embedding_1_embeddings_momentum:	H
4assignvariableop_57_rmsprop_gru_cell_kernel_momentum:
¬Q
>assignvariableop_58_rmsprop_gru_cell_recurrent_kernel_momentum:	d¬E
2assignvariableop_59_rmsprop_gru_cell_bias_momentum:	¬I
6assignvariableop_60_rmsprop_gru_cell_1_kernel_momentum:	d¬S
@assignvariableop_61_rmsprop_gru_cell_1_recurrent_kernel_momentum:	d¬G
4assignvariableop_62_rmsprop_gru_cell_1_bias_momentum:	¬E
3assignvariableop_63_rmsprop_dense_3_kernel_momentum:d@?
1assignvariableop_64_rmsprop_dense_3_bias_momentum:@E
3assignvariableop_65_rmsprop_dense_4_kernel_momentum:@?
1assignvariableop_66_rmsprop_dense_4_bias_momentum:
identity_68¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ù 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*ÿ
valueõBòDB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/16/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/17/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHû
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B õ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp(assignvariableop_10_embedding_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp*assignvariableop_11_embedding_1_embeddingsIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_gru_cell_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp-assignvariableop_13_gru_cell_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp!assignvariableop_14_gru_cell_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp%assignvariableop_15_gru_cell_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_16AssignVariableOp/assignvariableop_16_gru_cell_1_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp#assignvariableop_17_gru_cell_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_22AssignVariableOp assignvariableop_22_rmsprop_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp!assignvariableop_23_rmsprop_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_rmsprop_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp$assignvariableop_25_rmsprop_momentumIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_rmsprop_rhoIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp,assignvariableop_27_rmsprop_dense_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp*assignvariableop_28_rmsprop_dense_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp.assignvariableop_29_rmsprop_dense_1_kernel_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp,assignvariableop_30_rmsprop_dense_1_bias_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp.assignvariableop_31_rmsprop_dense_2_kernel_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp,assignvariableop_32_rmsprop_dense_2_bias_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_33AssignVariableOp9assignvariableop_33_rmsprop_batch_normalization_gamma_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_34AssignVariableOp8assignvariableop_34_rmsprop_batch_normalization_beta_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_35AssignVariableOp4assignvariableop_35_rmsprop_embedding_embeddings_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_36AssignVariableOp6assignvariableop_36_rmsprop_embedding_1_embeddings_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_37AssignVariableOp/assignvariableop_37_rmsprop_gru_cell_kernel_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_38AssignVariableOp9assignvariableop_38_rmsprop_gru_cell_recurrent_kernel_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp-assignvariableop_39_rmsprop_gru_cell_bias_rmsIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_40AssignVariableOp1assignvariableop_40_rmsprop_gru_cell_1_kernel_rmsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_41AssignVariableOp;assignvariableop_41_rmsprop_gru_cell_1_recurrent_kernel_rmsIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_42AssignVariableOp/assignvariableop_42_rmsprop_gru_cell_1_bias_rmsIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp.assignvariableop_43_rmsprop_dense_3_kernel_rmsIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp,assignvariableop_44_rmsprop_dense_3_bias_rmsIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp.assignvariableop_45_rmsprop_dense_4_kernel_rmsIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp,assignvariableop_46_rmsprop_dense_4_bias_rmsIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_rmsprop_dense_kernel_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_48AssignVariableOp/assignvariableop_48_rmsprop_dense_bias_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_49AssignVariableOp3assignvariableop_49_rmsprop_dense_1_kernel_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_50AssignVariableOp1assignvariableop_50_rmsprop_dense_1_bias_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_51AssignVariableOp3assignvariableop_51_rmsprop_dense_2_kernel_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_52AssignVariableOp1assignvariableop_52_rmsprop_dense_2_bias_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_53AssignVariableOp>assignvariableop_53_rmsprop_batch_normalization_gamma_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_54AssignVariableOp=assignvariableop_54_rmsprop_batch_normalization_beta_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_55AssignVariableOp9assignvariableop_55_rmsprop_embedding_embeddings_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_56AssignVariableOp;assignvariableop_56_rmsprop_embedding_1_embeddings_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_57AssignVariableOp4assignvariableop_57_rmsprop_gru_cell_kernel_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_58AssignVariableOp>assignvariableop_58_rmsprop_gru_cell_recurrent_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_59AssignVariableOp2assignvariableop_59_rmsprop_gru_cell_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_60AssignVariableOp6assignvariableop_60_rmsprop_gru_cell_1_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_61AssignVariableOp@assignvariableop_61_rmsprop_gru_cell_1_recurrent_kernel_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_62AssignVariableOp4assignvariableop_62_rmsprop_gru_cell_1_bias_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_63AssignVariableOp3assignvariableop_63_rmsprop_dense_3_kernel_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_64AssignVariableOp1assignvariableop_64_rmsprop_dense_3_bias_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_65AssignVariableOp3assignvariableop_65_rmsprop_dense_4_kernel_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_66AssignVariableOp1assignvariableop_66_rmsprop_dense_4_bias_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_68IdentityIdentity_67:output:0^NoOp_1*
T0*
_output_shapes
: þ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_68Identity_68:output:0*
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Þ«
ï
__inference_call_337801363
input_0
input_1
input_2	
input_3
input_4A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@D
1sequential_dense_1_matmul_readvariableop_resource:	@A
2sequential_dense_1_biasadd_readvariableop_resource:	F
3sequential_1_dense_2_matmul_readvariableop_resource:	C
4sequential_1_dense_2_biasadd_readvariableop_resource:	L
=sequential_1_batch_normalization_cast_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_1_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_2_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_3_readvariableop_resource:	7
$embedding_embedding_lookup_337801253:	9
&embedding_1_embedding_lookup_337801260:	3
 gru_cell_readvariableop_resource:	¬;
'gru_cell_matmul_readvariableop_resource:
¬<
)gru_cell_matmul_1_readvariableop_resource:	d¬5
"gru_cell_1_readvariableop_resource:	¬<
)gru_cell_1_matmul_readvariableop_resource:	d¬>
+gru_cell_1_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_3_matmul_readvariableop_resource:d@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢embedding/embedding_lookup¢embedding_1/embedding_lookup¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢4sequential_1/batch_normalization/Cast/ReadVariableOp¢6sequential_1/batch_normalization/Cast_1/ReadVariableOp¢6sequential_1/batch_normalization/Cast_2/ReadVariableOp¢6sequential_1/batch_normalization/Cast_3/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOpc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          e
	transpose	Transposeinput_4transpose/perm:output:0*
T0*"
_output_shapes
:d
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
sequential/dense/MatMulMatMulinput_0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@i
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes

:@
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¤
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	n
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	M
CastCastinput_2*

DstT0*

SrcT0	*
_output_shapes

:
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential_1/dense_2/MatMulMatMulCast:y:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	r
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:	¯
4sequential_1/batch_normalization/Cast/ReadVariableOpReadVariableOp=sequential_1_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_2/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_3/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:*
dtype0u
0sequential_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ø
.sequential_1/batch_normalization/batchnorm/addAddV2>sequential_1/batch_normalization/Cast_1/ReadVariableOp:value:09sequential_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0sequential_1/batch_normalization/batchnorm/RsqrtRsqrt2sequential_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/mulMul4sequential_1/batch_normalization/batchnorm/Rsqrt:y:0>sequential_1/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:¾
0sequential_1/batch_normalization/batchnorm/mul_1Mul'sequential_1/dense_2/Relu:activations:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:	Ï
0sequential_1/batch_normalization/batchnorm/mul_2Mul<sequential_1/batch_normalization/Cast/ReadVariableOp:value:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/subSub>sequential_1/batch_normalization/Cast_2/ReadVariableOp:value:04sequential_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Í
0sequential_1/batch_normalization/batchnorm/add_1AddV24sequential_1/batch_normalization/batchnorm/mul_1:z:02sequential_1/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes
:	O
Cast_1Castinput_1*

DstT0*

SrcT0*
_output_shapes

:Z
embedding/CastCast
Cast_1:y:0*

DstT0*

SrcT0*
_output_shapes

:á
embedding/embedding_lookupResourceGather$embedding_embedding_lookup_337801253embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding/embedding_lookup/337801253*#
_output_shapes
:*
dtype0»
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding/embedding_lookup/337801253*#
_output_shapes
:
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
:O
Cast_2Castinput_3*

DstT0*

SrcT0*
_output_shapes

:\
embedding_1/CastCast
Cast_2:y:0*

DstT0*

SrcT0*
_output_shapes

:é
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_337801260embedding_1/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801260*#
_output_shapes
:*
dtype0Á
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801260*#
_output_shapes
:
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
:^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0Reshape/shape:output:0*
T0*
_output_shapes
:	`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape0embedding_1/embedding_lookup/Identity_1:output:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	s
add/addAddV2%sequential/dense_1/Relu:activations:0Reshape:output:0*
T0*
_output_shapes
:	
	add/add_1AddV2add/add:z:04sequential_1/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:	_
	add/add_2AddV2add/add_1:z:0Reshape_1:output:0*
T0*
_output_shapes
:	]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	¬*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0z
gru_cell/MatMulMatMuladd/add_2:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬{
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	¬c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:d:d:d*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell/MatMul_1MatMulstrided_slice:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	¬c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÏ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
:d:d:d*
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:dV
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:dt
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:dZ
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:do
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes

:dk
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes

:dR
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:dl
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0strided_slice:output:0*
T0*
_output_shapes

:dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:dc
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:dh
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:d_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMulMatMulgru_cell/add_3:z:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*
_output_shapes
:	¬e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*2
_output_shapes 
:d:d:d*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*
_output_shapes
:	¬e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ×
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
:d:d:d*
	num_splitx
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*
_output_shapes

:dZ
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes

:dz
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*
_output_shapes

:d^
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes

:du
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*
_output_shapes

:dq
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*
_output_shapes

:dV
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes

:dr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0strided_slice_1:output:0*
T0*
_output_shapes

:dU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?q
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes

:di
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes

:dn
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes

:d
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0
sequential_2/dense_3/MatMulMatMulgru_cell_1/add_3:z:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¬
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@q
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*
_output_shapes

:@
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0«
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:q
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*
_output_shapes

:m
IdentityIdentity'sequential_2/dense_4/Relu:activations:0^NoOp*
T0*
_output_shapes

:Z

Identity_1Identitygru_cell/add_3:z:0^NoOp*
T0*
_output_shapes

:d\

Identity_2Identitygru_cell_1/add_3:z:0^NoOp*
T0*
_output_shapes

:dá
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp5^sequential_1/batch_normalization/Cast/ReadVariableOp7^sequential_1/batch_normalization/Cast_1/ReadVariableOp7^sequential_1/batch_normalization/Cast_2/ReadVariableOp7^sequential_1/batch_normalization/Cast_3/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:::::d: : : : : : : : : : : : : : : : : : : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2l
4sequential_1/batch_normalization/Cast/ReadVariableOp4sequential_1/batch_normalization/Cast/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_1/ReadVariableOp6sequential_1/batch_normalization/Cast_1/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_2/ReadVariableOp6sequential_1/batch_normalization/Cast_2/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_3/ReadVariableOp6sequential_1/batch_normalization/Cast_3/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:G C

_output_shapes

:
!
_user_specified_name	input/0:GC

_output_shapes

:
!
_user_specified_name	input/1:GC

_output_shapes

:
!
_user_specified_name	input/2:GC

_output_shapes

:
!
_user_specified_name	input/3:KG
"
_output_shapes
:d
!
_user_specified_name	input/4
ä
Á
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799898
dense_2_input$
dense_2_337799883:	 
dense_2_337799885:	,
batch_normalization_337799888:	,
batch_normalization_337799890:	,
batch_normalization_337799892:	,
batch_normalization_337799894:	
identity¢+batch_normalization/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallý
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_337799883dense_2_337799885*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_337799752
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_337799888batch_normalization_337799890batch_normalization_337799892batch_normalization_337799894*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337799723
IdentityIdentity4batch_normalization/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_2_input
µ	
©
H__inference_embedding_layer_call_and_return_conditional_losses_337800103

inputs-
embedding_lookup_337800097:	
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
embedding_lookupResourceGatherembedding_lookup_337800097Cast:y:0*
Tindices0*-
_class#
!loc:@embedding_lookup/337800097*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¦
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/337800097*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Û
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337800362

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
Ö

B__inference_add_layer_call_and_return_conditional_losses_337800136

inputs
inputs_1
inputs_2
inputs_3
identityQ
addAddV2inputsinputs_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2add:z:0inputs_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_2AddV2	add_1:z:0inputs_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ«
ï
__inference_call_337801204
input_0
input_1	
input_2	
input_3
input_4A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@D
1sequential_dense_1_matmul_readvariableop_resource:	@A
2sequential_dense_1_biasadd_readvariableop_resource:	F
3sequential_1_dense_2_matmul_readvariableop_resource:	C
4sequential_1_dense_2_biasadd_readvariableop_resource:	L
=sequential_1_batch_normalization_cast_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_1_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_2_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_3_readvariableop_resource:	7
$embedding_embedding_lookup_337801094:	9
&embedding_1_embedding_lookup_337801101:	3
 gru_cell_readvariableop_resource:	¬;
'gru_cell_matmul_readvariableop_resource:
¬<
)gru_cell_matmul_1_readvariableop_resource:	d¬5
"gru_cell_1_readvariableop_resource:	¬<
)gru_cell_1_matmul_readvariableop_resource:	d¬>
+gru_cell_1_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_3_matmul_readvariableop_resource:d@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢embedding/embedding_lookup¢embedding_1/embedding_lookup¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢4sequential_1/batch_normalization/Cast/ReadVariableOp¢6sequential_1/batch_normalization/Cast_1/ReadVariableOp¢6sequential_1/batch_normalization/Cast_2/ReadVariableOp¢6sequential_1/batch_normalization/Cast_3/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOpc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          e
	transpose	Transposeinput_4transpose/perm:output:0*
T0*"
_output_shapes
:d
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
sequential/dense/MatMulMatMulinput_0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@i
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes

:@
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¤
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	n
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	M
CastCastinput_2*

DstT0*

SrcT0	*
_output_shapes

:
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential_1/dense_2/MatMulMatMulCast:y:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	r
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:	¯
4sequential_1/batch_normalization/Cast/ReadVariableOpReadVariableOp=sequential_1_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_2/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_3/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:*
dtype0u
0sequential_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ø
.sequential_1/batch_normalization/batchnorm/addAddV2>sequential_1/batch_normalization/Cast_1/ReadVariableOp:value:09sequential_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0sequential_1/batch_normalization/batchnorm/RsqrtRsqrt2sequential_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/mulMul4sequential_1/batch_normalization/batchnorm/Rsqrt:y:0>sequential_1/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:¾
0sequential_1/batch_normalization/batchnorm/mul_1Mul'sequential_1/dense_2/Relu:activations:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:	Ï
0sequential_1/batch_normalization/batchnorm/mul_2Mul<sequential_1/batch_normalization/Cast/ReadVariableOp:value:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/subSub>sequential_1/batch_normalization/Cast_2/ReadVariableOp:value:04sequential_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Í
0sequential_1/batch_normalization/batchnorm/add_1AddV24sequential_1/batch_normalization/batchnorm/mul_1:z:02sequential_1/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes
:	O
Cast_1Castinput_1*

DstT0*

SrcT0	*
_output_shapes

:Z
embedding/CastCast
Cast_1:y:0*

DstT0*

SrcT0*
_output_shapes

:á
embedding/embedding_lookupResourceGather$embedding_embedding_lookup_337801094embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding/embedding_lookup/337801094*#
_output_shapes
:*
dtype0»
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding/embedding_lookup/337801094*#
_output_shapes
:
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
:O
Cast_2Castinput_3*

DstT0*

SrcT0*
_output_shapes

:\
embedding_1/CastCast
Cast_2:y:0*

DstT0*

SrcT0*
_output_shapes

:é
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_337801101embedding_1/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801101*#
_output_shapes
:*
dtype0Á
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801101*#
_output_shapes
:
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
:^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0Reshape/shape:output:0*
T0*
_output_shapes
:	`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape0embedding_1/embedding_lookup/Identity_1:output:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	s
add/addAddV2%sequential/dense_1/Relu:activations:0Reshape:output:0*
T0*
_output_shapes
:	
	add/add_1AddV2add/add:z:04sequential_1/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:	_
	add/add_2AddV2add/add_1:z:0Reshape_1:output:0*
T0*
_output_shapes
:	]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	¬*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0z
gru_cell/MatMulMatMuladd/add_2:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬{
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	¬c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:d:d:d*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell/MatMul_1MatMulstrided_slice:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	¬c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÏ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
:d:d:d*
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:dV
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:dt
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:dZ
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:do
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes

:dk
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes

:dR
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:dl
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0strided_slice:output:0*
T0*
_output_shapes

:dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:dc
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:dh
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:d_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMulMatMulgru_cell/add_3:z:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*
_output_shapes
:	¬e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*2
_output_shapes 
:d:d:d*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*
_output_shapes
:	¬e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ×
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
:d:d:d*
	num_splitx
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*
_output_shapes

:dZ
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes

:dz
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*
_output_shapes

:d^
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes

:du
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*
_output_shapes

:dq
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*
_output_shapes

:dV
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes

:dr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0strided_slice_1:output:0*
T0*
_output_shapes

:dU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?q
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes

:di
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes

:dn
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes

:d
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0
sequential_2/dense_3/MatMulMatMulgru_cell_1/add_3:z:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¬
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@q
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*
_output_shapes

:@
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0«
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:q
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*
_output_shapes

:m
IdentityIdentity'sequential_2/dense_4/Relu:activations:0^NoOp*
T0*
_output_shapes

:Z

Identity_1Identitygru_cell/add_3:z:0^NoOp*
T0*
_output_shapes

:d\

Identity_2Identitygru_cell_1/add_3:z:0^NoOp*
T0*
_output_shapes

:dá
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp5^sequential_1/batch_normalization/Cast/ReadVariableOp7^sequential_1/batch_normalization/Cast_1/ReadVariableOp7^sequential_1/batch_normalization/Cast_2/ReadVariableOp7^sequential_1/batch_normalization/Cast_3/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:::::d: : : : : : : : : : : : : : : : : : : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2l
4sequential_1/batch_normalization/Cast/ReadVariableOp4sequential_1/batch_normalization/Cast/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_1/ReadVariableOp6sequential_1/batch_normalization/Cast_1/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_2/ReadVariableOp6sequential_1/batch_normalization/Cast_2/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_3/ReadVariableOp6sequential_1/batch_normalization/Cast_3/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:G C

_output_shapes

:
!
_user_specified_name	input/0:GC

_output_shapes

:
!
_user_specified_name	input/1:GC

_output_shapes

:
!
_user_specified_name	input/2:GC

_output_shapes

:
!
_user_specified_name	input/3:KG
"
_output_shapes
:d
!
_user_specified_name	input/4
±

Ú
,__inference_gru_cell_layer_call_fn_337802589

inputs

states
unknown:	¬
	unknown_0:
¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatesunknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_layer_call_and_return_conditional_losses_337800430o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
²

Û
.__inference_gru_cell_1_layer_call_fn_337802681

inputs

states
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatesunknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337800233o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
¤
Ó
.__inference_sequential_layer_call_fn_337802358

inputs
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_337799540p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

´
I__inference_sequential_layer_call_and_return_conditional_losses_337799540

inputs!
dense_337799517:@
dense_337799519:@$
dense_1_337799534:	@ 
dense_1_337799536:	
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallí
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_337799517dense_337799519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_337799516
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_337799534dense_1_337799536*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_337799533x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
ª
(__inference_rial_layer_call_fn_337802013
input_0
input_1
input_2
input_3
input_4
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	¬

unknown_12:
¬

unknown_13:	d¬

unknown_14:	¬

unknown_15:	d¬

unknown_16:	d¬

unknown_17:d@

unknown_18:@

unknown_19:@

unknown_20:
identity

identity_1

identity_2¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinput_0input_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*&
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_rial_layer_call_and_return_conditional_losses_337800610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/3:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input/4
Ê

+__inference_dense_1_layer_call_fn_337802864

inputs
unknown:	@
	unknown_0:	
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_337799533p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ

+__inference_dense_3_layer_call_fn_337802984

inputs
unknown:d@
	unknown_0:@
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_337799916o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¥
¼
"__inference__traced_save_337803245
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop3
/savev2_embedding_embeddings_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop.
*savev2_gru_cell_kernel_read_readvariableop8
4savev2_gru_cell_recurrent_kernel_read_readvariableop,
(savev2_gru_cell_bias_read_readvariableop0
,savev2_gru_cell_1_kernel_read_readvariableop:
6savev2_gru_cell_1_recurrent_kernel_read_readvariableop.
*savev2_gru_cell_1_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_1_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_2_bias_rms_read_readvariableopD
@savev2_rmsprop_batch_normalization_gamma_rms_read_readvariableopC
?savev2_rmsprop_batch_normalization_beta_rms_read_readvariableop?
;savev2_rmsprop_embedding_embeddings_rms_read_readvariableopA
=savev2_rmsprop_embedding_1_embeddings_rms_read_readvariableop:
6savev2_rmsprop_gru_cell_kernel_rms_read_readvariableopD
@savev2_rmsprop_gru_cell_recurrent_kernel_rms_read_readvariableop8
4savev2_rmsprop_gru_cell_bias_rms_read_readvariableop<
8savev2_rmsprop_gru_cell_1_kernel_rms_read_readvariableopF
Bsavev2_rmsprop_gru_cell_1_recurrent_kernel_rms_read_readvariableop:
6savev2_rmsprop_gru_cell_1_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_3_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_4_bias_rms_read_readvariableop<
8savev2_rmsprop_dense_kernel_momentum_read_readvariableop:
6savev2_rmsprop_dense_bias_momentum_read_readvariableop>
:savev2_rmsprop_dense_1_kernel_momentum_read_readvariableop<
8savev2_rmsprop_dense_1_bias_momentum_read_readvariableop>
:savev2_rmsprop_dense_2_kernel_momentum_read_readvariableop<
8savev2_rmsprop_dense_2_bias_momentum_read_readvariableopI
Esavev2_rmsprop_batch_normalization_gamma_momentum_read_readvariableopH
Dsavev2_rmsprop_batch_normalization_beta_momentum_read_readvariableopD
@savev2_rmsprop_embedding_embeddings_momentum_read_readvariableopF
Bsavev2_rmsprop_embedding_1_embeddings_momentum_read_readvariableop?
;savev2_rmsprop_gru_cell_kernel_momentum_read_readvariableopI
Esavev2_rmsprop_gru_cell_recurrent_kernel_momentum_read_readvariableop=
9savev2_rmsprop_gru_cell_bias_momentum_read_readvariableopA
=savev2_rmsprop_gru_cell_1_kernel_momentum_read_readvariableopK
Gsavev2_rmsprop_gru_cell_1_recurrent_kernel_momentum_read_readvariableop?
;savev2_rmsprop_gru_cell_1_bias_momentum_read_readvariableop>
:savev2_rmsprop_dense_3_kernel_momentum_read_readvariableop<
8savev2_rmsprop_dense_3_bias_momentum_read_readvariableop>
:savev2_rmsprop_dense_4_kernel_momentum_read_readvariableop<
8savev2_rmsprop_dense_4_bias_momentum_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ö 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*ÿ
valueõBòDB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/16/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/17/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/18/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/19/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHø
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ´
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop/savev2_embedding_embeddings_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop*savev2_gru_cell_kernel_read_readvariableop4savev2_gru_cell_recurrent_kernel_read_readvariableop(savev2_gru_cell_bias_read_readvariableop,savev2_gru_cell_1_kernel_read_readvariableop6savev2_gru_cell_1_recurrent_kernel_read_readvariableop*savev2_gru_cell_1_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableop5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop3savev2_rmsprop_dense_1_bias_rms_read_readvariableop5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop3savev2_rmsprop_dense_2_bias_rms_read_readvariableop@savev2_rmsprop_batch_normalization_gamma_rms_read_readvariableop?savev2_rmsprop_batch_normalization_beta_rms_read_readvariableop;savev2_rmsprop_embedding_embeddings_rms_read_readvariableop=savev2_rmsprop_embedding_1_embeddings_rms_read_readvariableop6savev2_rmsprop_gru_cell_kernel_rms_read_readvariableop@savev2_rmsprop_gru_cell_recurrent_kernel_rms_read_readvariableop4savev2_rmsprop_gru_cell_bias_rms_read_readvariableop8savev2_rmsprop_gru_cell_1_kernel_rms_read_readvariableopBsavev2_rmsprop_gru_cell_1_recurrent_kernel_rms_read_readvariableop6savev2_rmsprop_gru_cell_1_bias_rms_read_readvariableop5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop3savev2_rmsprop_dense_3_bias_rms_read_readvariableop5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop3savev2_rmsprop_dense_4_bias_rms_read_readvariableop8savev2_rmsprop_dense_kernel_momentum_read_readvariableop6savev2_rmsprop_dense_bias_momentum_read_readvariableop:savev2_rmsprop_dense_1_kernel_momentum_read_readvariableop8savev2_rmsprop_dense_1_bias_momentum_read_readvariableop:savev2_rmsprop_dense_2_kernel_momentum_read_readvariableop8savev2_rmsprop_dense_2_bias_momentum_read_readvariableopEsavev2_rmsprop_batch_normalization_gamma_momentum_read_readvariableopDsavev2_rmsprop_batch_normalization_beta_momentum_read_readvariableop@savev2_rmsprop_embedding_embeddings_momentum_read_readvariableopBsavev2_rmsprop_embedding_1_embeddings_momentum_read_readvariableop;savev2_rmsprop_gru_cell_kernel_momentum_read_readvariableopEsavev2_rmsprop_gru_cell_recurrent_kernel_momentum_read_readvariableop9savev2_rmsprop_gru_cell_bias_momentum_read_readvariableop=savev2_rmsprop_gru_cell_1_kernel_momentum_read_readvariableopGsavev2_rmsprop_gru_cell_1_recurrent_kernel_momentum_read_readvariableop;savev2_rmsprop_gru_cell_1_bias_momentum_read_readvariableop:savev2_rmsprop_dense_3_kernel_momentum_read_readvariableop8savev2_rmsprop_dense_3_bias_momentum_read_readvariableop:savev2_rmsprop_dense_4_kernel_momentum_read_readvariableop8savev2_rmsprop_dense_4_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*â
_input_shapesÐ
Í: :@:@:	@::	::::::	:	:
¬:	d¬:	¬:	d¬:	d¬:	¬:d@:@:@:: : : : : :@:@:	@::	::::	:	:
¬:	d¬:	¬:	d¬:	d¬:	¬:d@:@:@::@:@:	@::	::::	:	:
¬:	d¬:	¬:	d¬:	d¬:	¬:d@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::%!

_output_shapes
:	:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::%!

_output_shapes
:	:%!

_output_shapes
:	:&"
 
_output_shapes
:
¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	¬:$ 

_output_shapes

:d@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::% !

_output_shapes
:	:!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::%$!

_output_shapes
:	:%%!

_output_shapes
:	:&&"
 
_output_shapes
:
¬:%'!

_output_shapes
:	d¬:%(!

_output_shapes
:	¬:%)!

_output_shapes
:	d¬:%*!

_output_shapes
:	d¬:%+!

_output_shapes
:	¬:$, 

_output_shapes

:d@: -

_output_shapes
:@:$. 

_output_shapes

:@: /

_output_shapes
::$0 

_output_shapes

:@: 1

_output_shapes
:@:%2!

_output_shapes
:	@:!3

_output_shapes	
::%4!

_output_shapes
:	:!5

_output_shapes	
::!6

_output_shapes	
::!7

_output_shapes	
::%8!

_output_shapes
:	:%9!

_output_shapes
:	:&:"
 
_output_shapes
:
¬:%;!

_output_shapes
:	d¬:%<!

_output_shapes
:	¬:%=!

_output_shapes
:	d¬:%>!

_output_shapes
:	d¬:%?!

_output_shapes
:	¬:$@ 

_output_shapes

:d@: A

_output_shapes
:@:$B 

_output_shapes

:@: C

_output_shapes
::D

_output_shapes
: 
²
Ö
7__inference_batch_normalization_layer_call_fn_337802908

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337799676p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

0__inference_sequential_1_layer_call_fn_337799862
dense_2_input
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799830p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namedense_2_input
±

/__inference_embedding_1_layer_call_fn_337802533

inputs
unknown:	
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_337800118t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
E
¥

C__inference_rial_layer_call_and_return_conditional_losses_337800610	
input
input_1
input_2
input_3
input_4&
sequential_337800538:@"
sequential_337800540:@'
sequential_337800542:	@#
sequential_337800544:	)
sequential_1_337800548:	%
sequential_1_337800550:	%
sequential_1_337800552:	%
sequential_1_337800554:	%
sequential_1_337800556:	%
sequential_1_337800558:	&
embedding_337800562:	(
embedding_1_337800566:	%
gru_cell_337800578:	¬&
gru_cell_337800580:
¬%
gru_cell_337800582:	d¬'
gru_cell_1_337800590:	¬'
gru_cell_1_337800592:	d¬'
gru_cell_1_337800594:	d¬(
sequential_2_337800598:d@$
sequential_2_337800600:@(
sequential_2_337800602:@$
sequential_2_337800604:
identity

identity_1

identity_2¢!embedding/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢ gru_cell/StatefulPartitionedCall¢"gru_cell_1/StatefulPartitionedCall¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_2/StatefulPartitionedCallc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinput_4transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd±
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputsequential_337800538sequential_337800540sequential_337800542sequential_337800544*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_337799600V
CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallCast:y:0sequential_1_337800548sequential_1_337800550sequential_1_337800552sequential_1_337800554sequential_1_337800556sequential_1_337800558*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799830X
Cast_1Castinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
!embedding/StatefulPartitionedCallStatefulPartitionedCall
Cast_1:y:0embedding_337800562*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_layer_call_and_return_conditional_losses_337800103X
Cast_2Castinput_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall
Cast_2:y:0embedding_1_337800566*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_337800118^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape*embedding/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape,embedding_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
add/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0Reshape:output:0-sequential_1/StatefulPartitionedCall:output:0Reshape_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_layer_call_and_return_conditional_losses_337800136]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskÒ
 gru_cell/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0strided_slice:output:0gru_cell_337800578gru_cell_337800580gru_cell_337800582*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_layer_call_and_return_conditional_losses_337800430_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskë
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall)gru_cell/StatefulPartitionedCall:output:0strided_slice_1:output:0gru_cell_1_337800590gru_cell_1_337800592gru_cell_1_337800594*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337800362â
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall+gru_cell_1/StatefulPartitionedCall:output:0sequential_2_337800598sequential_2_337800600sequential_2_337800602sequential_2_337800604*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800000|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz

Identity_1Identity)gru_cell/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|

Identity_2Identity+gru_cell_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdË
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall!^gru_cell/StatefulPartitionedCall#^gru_cell_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:RN
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

_user_specified_nameinput
E
¥

C__inference_rial_layer_call_and_return_conditional_losses_337800254	
input
input_1
input_2
input_3
input_4&
sequential_337800069:@"
sequential_337800071:@'
sequential_337800073:	@#
sequential_337800075:	)
sequential_1_337800079:	%
sequential_1_337800081:	%
sequential_1_337800083:	%
sequential_1_337800085:	%
sequential_1_337800087:	%
sequential_1_337800089:	&
embedding_337800104:	(
embedding_1_337800119:	%
gru_cell_337800182:	¬&
gru_cell_337800184:
¬%
gru_cell_337800186:	d¬'
gru_cell_1_337800234:	¬'
gru_cell_1_337800236:	d¬'
gru_cell_1_337800238:	d¬(
sequential_2_337800242:d@$
sequential_2_337800244:@(
sequential_2_337800246:@$
sequential_2_337800248:
identity

identity_1

identity_2¢!embedding/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢ gru_cell/StatefulPartitionedCall¢"gru_cell_1/StatefulPartitionedCall¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_2/StatefulPartitionedCallc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinput_4transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd±
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputsequential_337800069sequential_337800071sequential_337800073sequential_337800075*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_337799540V
CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallCast:y:0sequential_1_337800079sequential_1_337800081sequential_1_337800083sequential_1_337800085sequential_1_337800087sequential_1_337800089*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799768X
Cast_1Castinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
!embedding/StatefulPartitionedCallStatefulPartitionedCall
Cast_1:y:0embedding_337800104*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_layer_call_and_return_conditional_losses_337800103X
Cast_2Castinput_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall
Cast_2:y:0embedding_1_337800119*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_337800118^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape*embedding/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape,embedding_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
add/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0Reshape:output:0-sequential_1/StatefulPartitionedCall:output:0Reshape_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_layer_call_and_return_conditional_losses_337800136]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskÒ
 gru_cell/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0strided_slice:output:0gru_cell_337800182gru_cell_337800184gru_cell_337800186*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_layer_call_and_return_conditional_losses_337800181_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskë
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall)gru_cell/StatefulPartitionedCall:output:0strided_slice_1:output:0gru_cell_1_337800234gru_cell_1_337800236gru_cell_1_337800238*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337800233â
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall+gru_cell_1/StatefulPartitionedCall:output:0sequential_2_337800242sequential_2_337800244sequential_2_337800246sequential_2_337800248*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_337799940|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz

Identity_1Identity)gru_cell/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|

Identity_2Identity+gru_cell_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdË
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall!^gru_cell/StatefulPartitionedCall#^gru_cell_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:RN
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

_user_specified_nameinput
±

R__inference_batch_normalization_layer_call_and_return_conditional_losses_337799676

inputs+
cast_readvariableop_resource:	-
cast_1_readvariableop_resource:	-
cast_2_readvariableop_resource:	-
cast_3_readvariableop_resource:	
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
ï
__inference_call_337801840
input_0
input_1
input_2
input_3
input_4A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@D
1sequential_dense_1_matmul_readvariableop_resource:	@A
2sequential_dense_1_biasadd_readvariableop_resource:	F
3sequential_1_dense_2_matmul_readvariableop_resource:	C
4sequential_1_dense_2_biasadd_readvariableop_resource:	L
=sequential_1_batch_normalization_cast_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_1_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_2_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_3_readvariableop_resource:	7
$embedding_embedding_lookup_337801730:	9
&embedding_1_embedding_lookup_337801737:	3
 gru_cell_readvariableop_resource:	¬;
'gru_cell_matmul_readvariableop_resource:
¬<
)gru_cell_matmul_1_readvariableop_resource:	d¬5
"gru_cell_1_readvariableop_resource:	¬<
)gru_cell_1_matmul_readvariableop_resource:	d¬>
+gru_cell_1_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_3_matmul_readvariableop_resource:d@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢embedding/embedding_lookup¢embedding_1/embedding_lookup¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢4sequential_1/batch_normalization/Cast/ReadVariableOp¢6sequential_1/batch_normalization/Cast_1/ReadVariableOp¢6sequential_1/batch_normalization/Cast_2/ReadVariableOp¢6sequential_1/batch_normalization/Cast_3/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOpc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinput_4transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
sequential/dense/MatMulMatMulinput_0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0­
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential_1/dense_2/MatMulMatMulCast:y:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¶
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
4sequential_1/batch_normalization/Cast/ReadVariableOpReadVariableOp=sequential_1_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_2/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_3/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:*
dtype0u
0sequential_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ø
.sequential_1/batch_normalization/batchnorm/addAddV2>sequential_1/batch_normalization/Cast_1/ReadVariableOp:value:09sequential_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0sequential_1/batch_normalization/batchnorm/RsqrtRsqrt2sequential_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/mulMul4sequential_1/batch_normalization/batchnorm/Rsqrt:y:0>sequential_1/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ç
0sequential_1/batch_normalization/batchnorm/mul_1Mul'sequential_1/dense_2/Relu:activations:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
0sequential_1/batch_normalization/batchnorm/mul_2Mul<sequential_1/batch_normalization/Cast/ReadVariableOp:value:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/subSub>sequential_1/batch_normalization/Cast_2/ReadVariableOp:value:04sequential_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ö
0sequential_1/batch_normalization/batchnorm/add_1AddV24sequential_1/batch_normalization/batchnorm/mul_1:z:02sequential_1/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Cast_1Castinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
embedding/CastCast
Cast_1:y:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
embedding/embedding_lookupResourceGather$embedding_embedding_lookup_337801730embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding/embedding_lookup/337801730*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ä
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding/embedding_lookup/337801730*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Cast_2Castinput_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
embedding_1/CastCast
Cast_2:y:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_337801737embedding_1/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801737*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ê
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801737*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape0embedding_1/embedding_lookup/Identity_1:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
add/addAddV2%sequential/dense_1/Relu:activations:0Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
	add/add_1AddV2add/add:z:04sequential_1/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
	add/add_2AddV2add/add_1:z:0Reshape_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	¬*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0
gru_cell/MatMulMatMuladd/add_2:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell/MatMul_1MatMulstrided_slice:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMulMatMulgru_cell/add_3:z:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0¡
sequential_2/dense_3/MatMulMatMulgru_cell_1/add_3:z:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0´
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_2/dense_4/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc

Identity_1Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde

Identity_2Identitygru_cell_1/add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdá
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp5^sequential_1/batch_normalization/Cast/ReadVariableOp7^sequential_1/batch_normalization/Cast_1/ReadVariableOp7^sequential_1/batch_normalization/Cast_2/ReadVariableOp7^sequential_1/batch_normalization/Cast_3/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2l
4sequential_1/batch_normalization/Cast/ReadVariableOp4sequential_1/batch_normalization/Cast/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_1/ReadVariableOp6sequential_1/batch_normalization/Cast_1/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_2/ReadVariableOp6sequential_1/batch_normalization/Cast_2/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_3/ReadVariableOp6sequential_1/batch_normalization/Cast_3/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/0:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input/3:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input/4
Æ

+__inference_dense_4_layer_call_fn_337803004

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_337799933o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¥

ù
F__inference_dense_1_layer_call_and_return_conditional_losses_337802875

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤
Ó
0__inference_sequential_2_layer_call_fn_337802799

inputs
unknown:d@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800000o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
°
Ö
7__inference_batch_normalization_layer_call_fn_337802921

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337799723p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
Ø
.__inference_sequential_layer_call_fn_337799551
dense_input
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_337799540p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
¢
¹
I__inference_sequential_layer_call_and_return_conditional_losses_337799638
dense_input!
dense_337799627:@
dense_337799629:@$
dense_1_337799632:	@ 
dense_1_337799634:	
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallò
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_337799627dense_337799629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_337799516
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_337799632dense_1_337799634*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_337799533x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
²
í
__inference_call_337799447	
input
input_1
input_2
input_3
input_4A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@D
1sequential_dense_1_matmul_readvariableop_resource:	@A
2sequential_dense_1_biasadd_readvariableop_resource:	F
3sequential_1_dense_2_matmul_readvariableop_resource:	C
4sequential_1_dense_2_biasadd_readvariableop_resource:	L
=sequential_1_batch_normalization_cast_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_1_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_2_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_3_readvariableop_resource:	7
$embedding_embedding_lookup_337799337:	9
&embedding_1_embedding_lookup_337799344:	3
 gru_cell_readvariableop_resource:	¬;
'gru_cell_matmul_readvariableop_resource:
¬<
)gru_cell_matmul_1_readvariableop_resource:	d¬5
"gru_cell_1_readvariableop_resource:	¬<
)gru_cell_1_matmul_readvariableop_resource:	d¬>
+gru_cell_1_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_3_matmul_readvariableop_resource:d@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢embedding/embedding_lookup¢embedding_1/embedding_lookup¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢4sequential_1/batch_normalization/Cast/ReadVariableOp¢6sequential_1/batch_normalization/Cast_1/ReadVariableOp¢6sequential_1/batch_normalization/Cast_2/ReadVariableOp¢6sequential_1/batch_normalization/Cast_3/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOpc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinput_4transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
sequential/dense/MatMulMatMulinput.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0­
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential_1/dense_2/MatMulMatMulCast:y:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¶
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
4sequential_1/batch_normalization/Cast/ReadVariableOpReadVariableOp=sequential_1_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_2/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_3/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:*
dtype0u
0sequential_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ø
.sequential_1/batch_normalization/batchnorm/addAddV2>sequential_1/batch_normalization/Cast_1/ReadVariableOp:value:09sequential_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0sequential_1/batch_normalization/batchnorm/RsqrtRsqrt2sequential_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/mulMul4sequential_1/batch_normalization/batchnorm/Rsqrt:y:0>sequential_1/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ç
0sequential_1/batch_normalization/batchnorm/mul_1Mul'sequential_1/dense_2/Relu:activations:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
0sequential_1/batch_normalization/batchnorm/mul_2Mul<sequential_1/batch_normalization/Cast/ReadVariableOp:value:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/subSub>sequential_1/batch_normalization/Cast_2/ReadVariableOp:value:04sequential_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ö
0sequential_1/batch_normalization/batchnorm/add_1AddV24sequential_1/batch_normalization/batchnorm/mul_1:z:02sequential_1/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Cast_1Castinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
embedding/CastCast
Cast_1:y:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
embedding/embedding_lookupResourceGather$embedding_embedding_lookup_337799337embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding/embedding_lookup/337799337*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ä
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding/embedding_lookup/337799337*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
Cast_2Castinput_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
embedding_1/CastCast
Cast_2:y:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_337799344embedding_1/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/337799344*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Ê
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/337799344*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape0embedding_1/embedding_lookup/Identity_1:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
add/addAddV2%sequential/dense_1/Relu:activations:0Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
	add/add_1AddV2add/add:z:04sequential_1/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
	add/add_2AddV2add/add_1:z:0Reshape_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	¬*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0
gru_cell/MatMulMatMuladd/add_2:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell/MatMul_1MatMulstrided_slice:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMulMatMulgru_cell/add_3:z:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0¡
sequential_2/dense_3/MatMulMatMulgru_cell_1/add_3:z:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0´
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_2/dense_4/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc

Identity_1Identitygru_cell/add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde

Identity_2Identitygru_cell_1/add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdá
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp5^sequential_1/batch_normalization/Cast/ReadVariableOp7^sequential_1/batch_normalization/Cast_1/ReadVariableOp7^sequential_1/batch_normalization/Cast_2/ReadVariableOp7^sequential_1/batch_normalization/Cast_3/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2l
4sequential_1/batch_normalization/Cast/ReadVariableOp4sequential_1/batch_normalization/Cast/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_1/ReadVariableOp6sequential_1/batch_normalization/Cast_1/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_2/ReadVariableOp6sequential_1/batch_normalization/Cast_2/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_3/ReadVariableOp6sequential_1/batch_normalization/Cast_3/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput:RN
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

_user_specified_nameinput
§
ª
(__inference_rial_layer_call_fn_337800718
input_1
input_2
input_3
input_4
input_5
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	¬

unknown_12:
¬

unknown_13:	d¬

unknown_14:	¬

unknown_15:	d¬

unknown_16:	d¬

unknown_17:d@

unknown_18:@

unknown_19:@

unknown_20:
identity

identity_1

identity_2¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*&
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_rial_layer_call_and_return_conditional_losses_337800610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_5
E
§

C__inference_rial_layer_call_and_return_conditional_losses_337800799
input_1
input_2
input_3
input_4
input_5&
sequential_337800727:@"
sequential_337800729:@'
sequential_337800731:	@#
sequential_337800733:	)
sequential_1_337800737:	%
sequential_1_337800739:	%
sequential_1_337800741:	%
sequential_1_337800743:	%
sequential_1_337800745:	%
sequential_1_337800747:	&
embedding_337800751:	(
embedding_1_337800755:	%
gru_cell_337800767:	¬&
gru_cell_337800769:
¬%
gru_cell_337800771:	d¬'
gru_cell_1_337800779:	¬'
gru_cell_1_337800781:	d¬'
gru_cell_1_337800783:	d¬(
sequential_2_337800787:d@$
sequential_2_337800789:@(
sequential_2_337800791:@$
sequential_2_337800793:
identity

identity_1

identity_2¢!embedding/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢ gru_cell/StatefulPartitionedCall¢"gru_cell_1/StatefulPartitionedCall¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_2/StatefulPartitionedCallc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinput_5transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd³
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_337800727sequential_337800729sequential_337800731sequential_337800733*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_337799540V
CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallCast:y:0sequential_1_337800737sequential_1_337800739sequential_1_337800741sequential_1_337800743sequential_1_337800745sequential_1_337800747*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799768X
Cast_1Castinput_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
!embedding/StatefulPartitionedCallStatefulPartitionedCall
Cast_1:y:0embedding_337800751*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_layer_call_and_return_conditional_losses_337800103X
Cast_2Castinput_4*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall
Cast_2:y:0embedding_1_337800755*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_337800118^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape*embedding/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape,embedding_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
add/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0Reshape:output:0-sequential_1/StatefulPartitionedCall:output:0Reshape_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_add_layer_call_and_return_conditional_losses_337800136]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:á
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskÒ
 gru_cell/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0strided_slice:output:0gru_cell_337800767gru_cell_337800769gru_cell_337800771*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_gru_cell_layer_call_and_return_conditional_losses_337800181_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskë
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall)gru_cell/StatefulPartitionedCall:output:0strided_slice_1:output:0gru_cell_1_337800779gru_cell_1_337800781gru_cell_1_337800783*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337800233â
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall+gru_cell_1/StatefulPartitionedCall:output:0sequential_2_337800787sequential_2_337800789sequential_2_337800791sequential_2_337800793*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_337799940|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz

Identity_1Identity)gru_cell/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|

Identity_2Identity+gru_cell_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdË
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall!^gru_cell/StatefulPartitionedCall#^gru_cell_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_5
¥

ù
F__inference_dense_1_layer_call_and_return_conditional_losses_337799533

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ«
ï
__inference_call_337801681
input_0
input_1	
input_2	
input_3
input_4A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@D
1sequential_dense_1_matmul_readvariableop_resource:	@A
2sequential_dense_1_biasadd_readvariableop_resource:	F
3sequential_1_dense_2_matmul_readvariableop_resource:	C
4sequential_1_dense_2_biasadd_readvariableop_resource:	L
=sequential_1_batch_normalization_cast_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_1_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_2_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_3_readvariableop_resource:	7
$embedding_embedding_lookup_337801571:	9
&embedding_1_embedding_lookup_337801578:	3
 gru_cell_readvariableop_resource:	¬;
'gru_cell_matmul_readvariableop_resource:
¬<
)gru_cell_matmul_1_readvariableop_resource:	d¬5
"gru_cell_1_readvariableop_resource:	¬<
)gru_cell_1_matmul_readvariableop_resource:	d¬>
+gru_cell_1_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_3_matmul_readvariableop_resource:d@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢embedding/embedding_lookup¢embedding_1/embedding_lookup¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢4sequential_1/batch_normalization/Cast/ReadVariableOp¢6sequential_1/batch_normalization/Cast_1/ReadVariableOp¢6sequential_1/batch_normalization/Cast_2/ReadVariableOp¢6sequential_1/batch_normalization/Cast_3/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOpc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          e
	transpose	Transposeinput_4transpose/perm:output:0*
T0*"
_output_shapes
: d
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
sequential/dense/MatMulMatMulinput_0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: @
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: @i
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes

: @
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¤
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	 
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	 n
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	 M
CastCastinput_2*

DstT0*

SrcT0	*
_output_shapes

: 
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential_1/dense_2/MatMulMatMulCast:y:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	 
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	 r
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:	 ¯
4sequential_1/batch_normalization/Cast/ReadVariableOpReadVariableOp=sequential_1_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_2/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_3/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:*
dtype0u
0sequential_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ø
.sequential_1/batch_normalization/batchnorm/addAddV2>sequential_1/batch_normalization/Cast_1/ReadVariableOp:value:09sequential_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0sequential_1/batch_normalization/batchnorm/RsqrtRsqrt2sequential_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/mulMul4sequential_1/batch_normalization/batchnorm/Rsqrt:y:0>sequential_1/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:¾
0sequential_1/batch_normalization/batchnorm/mul_1Mul'sequential_1/dense_2/Relu:activations:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:	 Ï
0sequential_1/batch_normalization/batchnorm/mul_2Mul<sequential_1/batch_normalization/Cast/ReadVariableOp:value:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/subSub>sequential_1/batch_normalization/Cast_2/ReadVariableOp:value:04sequential_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Í
0sequential_1/batch_normalization/batchnorm/add_1AddV24sequential_1/batch_normalization/batchnorm/mul_1:z:02sequential_1/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes
:	 O
Cast_1Castinput_1*

DstT0*

SrcT0	*
_output_shapes

: Z
embedding/CastCast
Cast_1:y:0*

DstT0*

SrcT0*
_output_shapes

: á
embedding/embedding_lookupResourceGather$embedding_embedding_lookup_337801571embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding/embedding_lookup/337801571*#
_output_shapes
: *
dtype0»
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding/embedding_lookup/337801571*#
_output_shapes
: 
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
: O
Cast_2Castinput_3*

DstT0*

SrcT0*
_output_shapes

: \
embedding_1/CastCast
Cast_2:y:0*

DstT0*

SrcT0*
_output_shapes

: é
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_337801578embedding_1/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801578*#
_output_shapes
: *
dtype0Á
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/337801578*#
_output_shapes
: 
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
: ^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0Reshape/shape:output:0*
T0*
_output_shapes
:	 `
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape0embedding_1/embedding_lookup/Identity_1:output:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	 s
add/addAddV2%sequential/dense_1/Relu:activations:0Reshape:output:0*
T0*
_output_shapes
:	 
	add/add_1AddV2add/add:z:04sequential_1/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:	 _
	add/add_2AddV2add/add_1:z:0Reshape_1:output:0*
T0*
_output_shapes
:	 ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

: d*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	¬*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0z
gru_cell/MatMulMatMuladd/add_2:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ¬{
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	 ¬c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
: d: d: d*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell/MatMul_1MatMulstrided_slice:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ¬
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	 ¬c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÏ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
: d: d: d*
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

: dV
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

: dt
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

: dZ
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

: do
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes

: dk
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes

: dR
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

: dl
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0strided_slice:output:0*
T0*
_output_shapes

: dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

: dc
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

: dh
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

: d_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: d*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMulMatMulgru_cell/add_3:z:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ¬
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*
_output_shapes
:	 ¬e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*2
_output_shapes 
: d: d: d*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ¬
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*
_output_shapes
:	 ¬e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ×
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
: d: d: d*
	num_splitx
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*
_output_shapes

: dZ
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes

: dz
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*
_output_shapes

: d^
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes

: du
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*
_output_shapes

: dq
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*
_output_shapes

: dV
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes

: dr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0strided_slice_1:output:0*
T0*
_output_shapes

: dU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?q
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes

: di
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes

: dn
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes

: d
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0
sequential_2/dense_3/MatMulMatMulgru_cell_1/add_3:z:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: @
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¬
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: @q
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*
_output_shapes

: @
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0«
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*
_output_shapes

: m
IdentityIdentity'sequential_2/dense_4/Relu:activations:0^NoOp*
T0*
_output_shapes

: Z

Identity_1Identitygru_cell/add_3:z:0^NoOp*
T0*
_output_shapes

: d\

Identity_2Identitygru_cell_1/add_3:z:0^NoOp*
T0*
_output_shapes

: dá
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp5^sequential_1/batch_normalization/Cast/ReadVariableOp7^sequential_1/batch_normalization/Cast_1/ReadVariableOp7^sequential_1/batch_normalization/Cast_2/ReadVariableOp7^sequential_1/batch_normalization/Cast_3/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b: : : : : d: : : : : : : : : : : : : : : : : : : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2l
4sequential_1/batch_normalization/Cast/ReadVariableOp4sequential_1/batch_normalization/Cast/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_1/ReadVariableOp6sequential_1/batch_normalization/Cast_1/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_2/ReadVariableOp6sequential_1/batch_normalization/Cast_2/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_3/ReadVariableOp6sequential_1/batch_normalization/Cast_3/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:G C

_output_shapes

: 
!
_user_specified_name	input/0:GC

_output_shapes

: 
!
_user_specified_name	input/1:GC

_output_shapes

: 
!
_user_specified_name	input/2:GC

_output_shapes

: 
!
_user_specified_name	input/3:KG
"
_output_shapes
: d
!
_user_specified_name	input/4

Û
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337802734

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
·	
«
J__inference_embedding_1_layer_call_and_return_conditional_losses_337802543

inputs-
embedding_lookup_337802537:	
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
embedding_lookupResourceGatherembedding_lookup_337802537Cast:y:0*
Tindices0*-
_class#
!loc:@embedding_lookup/337802537*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¦
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/337802537*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
¹
I__inference_sequential_layer_call_and_return_conditional_losses_337799652
dense_input!
dense_337799641:@
dense_337799643:@$
dense_1_337799646:	@ 
dense_1_337799648:	
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallò
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_337799641dense_337799643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_337799516
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_337799646dense_1_337799648*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_337799533x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
ø

0__inference_sequential_1_layer_call_fn_337802424

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799768p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_3_layer_call_and_return_conditional_losses_337802995

inputs0
matmul_readvariableop_resource:d@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


÷
F__inference_dense_4_layer_call_and_return_conditional_losses_337803015

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§
º
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800000

inputs#
dense_3_337799989:d@
dense_3_337799991:@#
dense_4_337799994:@
dense_4_337799996:
identity¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCallõ
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_337799989dense_3_337799991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_337799916
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_337799994dense_4_337799996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_337799933w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
­

-__inference_embedding_layer_call_fn_337802516

inputs
unknown:	
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_layer_call_and_return_conditional_losses_337800103t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Ú
0__inference_sequential_2_layer_call_fn_337799951
dense_3_input
unknown:d@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_3_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_2_layer_call_and_return_conditional_losses_337799940o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'
_user_specified_namedense_3_input
·
Æ
I__inference_sequential_layer_call_and_return_conditional_losses_337802389

inputs6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@9
&dense_1_matmul_readvariableop_resource:	@6
'dense_1_biasadd_readvariableop_resource:	
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·	
«
J__inference_embedding_1_layer_call_and_return_conditional_losses_337800118

inputs-
embedding_lookup_337800112:	
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
embedding_lookupResourceGatherembedding_lookup_337800112Cast:y:0*
Tindices0*-
_class#
!loc:@embedding_lookup/337800112*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¦
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/337800112*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥

ù
F__inference_dense_2_layer_call_and_return_conditional_losses_337802895

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
Î
K__inference_sequential_2_layer_call_and_return_conditional_losses_337802835

inputs8
&dense_3_matmul_readvariableop_resource:d@5
'dense_3_biasadd_readvariableop_resource:@8
&dense_4_matmul_readvariableop_resource:@5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0y
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_4/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Þ«
ï
__inference_call_337801045
input_0
input_1
input_2
input_3
input_4A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@D
1sequential_dense_1_matmul_readvariableop_resource:	@A
2sequential_dense_1_biasadd_readvariableop_resource:	F
3sequential_1_dense_2_matmul_readvariableop_resource:	C
4sequential_1_dense_2_biasadd_readvariableop_resource:	L
=sequential_1_batch_normalization_cast_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_1_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_2_readvariableop_resource:	N
?sequential_1_batch_normalization_cast_3_readvariableop_resource:	7
$embedding_embedding_lookup_337800935:	9
&embedding_1_embedding_lookup_337800942:	3
 gru_cell_readvariableop_resource:	¬;
'gru_cell_matmul_readvariableop_resource:
¬<
)gru_cell_matmul_1_readvariableop_resource:	d¬5
"gru_cell_1_readvariableop_resource:	¬<
)gru_cell_1_matmul_readvariableop_resource:	d¬>
+gru_cell_1_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_3_matmul_readvariableop_resource:d@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢embedding/embedding_lookup¢embedding_1/embedding_lookup¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢4sequential_1/batch_normalization/Cast/ReadVariableOp¢6sequential_1/batch_normalization/Cast_1/ReadVariableOp¢6sequential_1/batch_normalization/Cast_2/ReadVariableOp¢6sequential_1/batch_normalization/Cast_3/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOpc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          e
	transpose	Transposeinput_4transpose/perm:output:0*
T0*"
_output_shapes
:d
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
sequential/dense/MatMulMatMulinput_0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@i
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes

:@
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¤
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	n
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	M
CastCastinput_2*

DstT0*

SrcT0*
_output_shapes

:
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential_1/dense_2/MatMulMatMulCast:y:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	r
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:	¯
4sequential_1/batch_normalization/Cast/ReadVariableOpReadVariableOp=sequential_1_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_1/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_2/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:*
dtype0³
6sequential_1/batch_normalization/Cast_3/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:*
dtype0u
0sequential_1/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ø
.sequential_1/batch_normalization/batchnorm/addAddV2>sequential_1/batch_normalization/Cast_1/ReadVariableOp:value:09sequential_1/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
0sequential_1/batch_normalization/batchnorm/RsqrtRsqrt2sequential_1/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/mulMul4sequential_1/batch_normalization/batchnorm/Rsqrt:y:0>sequential_1/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:¾
0sequential_1/batch_normalization/batchnorm/mul_1Mul'sequential_1/dense_2/Relu:activations:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:	Ï
0sequential_1/batch_normalization/batchnorm/mul_2Mul<sequential_1/batch_normalization/Cast/ReadVariableOp:value:02sequential_1/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ñ
.sequential_1/batch_normalization/batchnorm/subSub>sequential_1/batch_normalization/Cast_2/ReadVariableOp:value:04sequential_1/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Í
0sequential_1/batch_normalization/batchnorm/add_1AddV24sequential_1/batch_normalization/batchnorm/mul_1:z:02sequential_1/batch_normalization/batchnorm/sub:z:0*
T0*
_output_shapes
:	O
Cast_1Castinput_1*

DstT0*

SrcT0*
_output_shapes

:Z
embedding/CastCast
Cast_1:y:0*

DstT0*

SrcT0*
_output_shapes

:á
embedding/embedding_lookupResourceGather$embedding_embedding_lookup_337800935embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding/embedding_lookup/337800935*#
_output_shapes
:*
dtype0»
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding/embedding_lookup/337800935*#
_output_shapes
:
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
:O
Cast_2Castinput_3*

DstT0*

SrcT0*
_output_shapes

:\
embedding_1/CastCast
Cast_2:y:0*

DstT0*

SrcT0*
_output_shapes

:é
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_337800942embedding_1/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/337800942*#
_output_shapes
:*
dtype0Á
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/337800942*#
_output_shapes
:
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*#
_output_shapes
:^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0Reshape/shape:output:0*
T0*
_output_shapes
:	`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
	Reshape_1Reshape0embedding_1/embedding_lookup/Identity_1:output:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	s
add/addAddV2%sequential/dense_1/Relu:activations:0Reshape:output:0*
T0*
_output_shapes
:	
	add/add_1AddV2add/add:z:04sequential_1/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:	_
	add/add_2AddV2add/add_1:z:0Reshape_1:output:0*
T0*
_output_shapes
:	]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
strided_sliceStridedSlicetranspose:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	¬*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0z
gru_cell/MatMulMatMuladd/add_2:z:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬{
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	¬c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*2
_output_shapes 
:d:d:d*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell/MatMul_1MatMulstrided_slice:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	¬c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÏ
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
:d:d:d*
	num_splitr
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes

:dV
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes

:dt
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes

:dZ
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes

:do
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes

:dk
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes

:dR
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes

:dl
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0strided_slice:output:0*
T0*
_output_shapes

:dS
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes

:dc
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes

:dh
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes

:d_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*
shrink_axis_mask}
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMulMatMulgru_cell/add_3:z:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*
_output_shapes
:	¬e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*2
_output_shapes 
:d:d:d*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_1/MatMul_1MatMulstrided_slice_1:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	¬
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*
_output_shapes
:	¬e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ×
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*2
_output_shapes 
:d:d:d*
	num_splitx
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*
_output_shapes

:dZ
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*
_output_shapes

:dz
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*
_output_shapes

:d^
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*
_output_shapes

:du
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*
_output_shapes

:dq
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*
_output_shapes

:dV
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*
_output_shapes

:dr
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0strided_slice_1:output:0*
T0*
_output_shapes

:dU
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?q
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*
_output_shapes

:di
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*
_output_shapes

:dn
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*
_output_shapes

:d
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0
sequential_2/dense_3/MatMulMatMulgru_cell_1/add_3:z:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¬
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@q
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*
_output_shapes

:@
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0«
sequential_2/dense_4/MatMulMatMul'sequential_2/dense_3/Relu:activations:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:q
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*
_output_shapes

:m
IdentityIdentity'sequential_2/dense_4/Relu:activations:0^NoOp*
T0*
_output_shapes

:Z

Identity_1Identitygru_cell/add_3:z:0^NoOp*
T0*
_output_shapes

:d\

Identity_2Identitygru_cell_1/add_3:z:0^NoOp*
T0*
_output_shapes

:dá
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp5^sequential_1/batch_normalization/Cast/ReadVariableOp7^sequential_1/batch_normalization/Cast_1/ReadVariableOp7^sequential_1/batch_normalization/Cast_2/ReadVariableOp7^sequential_1/batch_normalization/Cast_3/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:::::d: : : : : : : : : : : : : : : : : : : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2l
4sequential_1/batch_normalization/Cast/ReadVariableOp4sequential_1/batch_normalization/Cast/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_1/ReadVariableOp6sequential_1/batch_normalization/Cast_1/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_2/ReadVariableOp6sequential_1/batch_normalization/Cast_2/ReadVariableOp2p
6sequential_1/batch_normalization/Cast_3/ReadVariableOp6sequential_1/batch_normalization/Cast_3/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp:G C

_output_shapes

:
!
_user_specified_name	input/0:GC

_output_shapes

:
!
_user_specified_name	input/1:GC

_output_shapes

:
!
_user_specified_name	input/2:GC

_output_shapes

:
!
_user_specified_name	input/3:KG
"
_output_shapes
:d
!
_user_specified_name	input/4


÷
F__inference_dense_3_layer_call_and_return_conditional_losses_337799916

inputs0
matmul_readvariableop_resource:d@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
î

$__inference__wrapped_model_337799498
input_1
input_2
input_3
input_4
input_5 
rial_337799448:@
rial_337799450:@!
rial_337799452:	@
rial_337799454:	!
rial_337799456:	
rial_337799458:	
rial_337799460:	
rial_337799462:	
rial_337799464:	
rial_337799466:	!
rial_337799468:	!
rial_337799470:	!
rial_337799472:	¬"
rial_337799474:
¬!
rial_337799476:	d¬!
rial_337799478:	¬!
rial_337799480:	d¬!
rial_337799482:	d¬ 
rial_337799484:d@
rial_337799486:@ 
rial_337799488:@
rial_337799490:
identity

identity_1

identity_2¢rial/StatefulPartitionedCallù
rial/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5rial_337799448rial_337799450rial_337799452rial_337799454rial_337799456rial_337799458rial_337799460rial_337799462rial_337799464rial_337799466rial_337799468rial_337799470rial_337799472rial_337799474rial_337799476rial_337799478rial_337799480rial_337799482rial_337799484rial_337799486rial_337799488rial_337799490*&
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *#
fR
__inference_call_337799447t
IdentityIdentity%rial/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv

Identity_1Identity%rial/StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv

Identity_2Identity%rial/StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
NoOpNoOp^rial/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : 2<
rial/StatefulPartitionedCallrial/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	input_5
Ï
º
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799830

inputs$
dense_2_337799815:	 
dense_2_337799817:	,
batch_normalization_337799820:	,
batch_normalization_337799822:	,
batch_normalization_337799824:	,
batch_normalization_337799826:	
identity¢+batch_normalization/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallö
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_337799815dense_2_337799817*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_337799752
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_337799820batch_normalization_337799822batch_normalization_337799824batch_normalization_337799826*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337799723
IdentityIdentity4batch_normalization/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ú
G__inference_gru_cell_layer_call_and_return_conditional_losses_337800430

inputs

states*
readvariableop_resource:	¬2
matmul_readvariableop_resource:
¬3
 matmul_1_readvariableop_resource:	d¬
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
ö

0__inference_sequential_1_layer_call_fn_337802441

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799830p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
Ø
.__inference_sequential_layer_call_fn_337799624
dense_input
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:	
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_layer_call_and_return_conditional_losses_337799600p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
;
input_20
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ
;
input_30
serving_default_input_3:0ÿÿÿÿÿÿÿÿÿ
;
input_40
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿ
?
input_54
serving_default_input_5:0ÿÿÿÿÿÿÿÿÿd<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ<
output_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿd<
output_30
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿdtensorflow/serving/predict:¤Þ
Ì
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
mlp
	mlp2

emb_act
emb_ind
add

concat
rnn1
rnn2
	q_net
	optimizer
call

signatures"
_tf_keras_model
Æ
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21"
trackable_list_wrapper
¶
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15
&16
'17
(18
)19"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
È
/trace_0
0trace_1
1trace_2
2trace_32Ý
(__inference_rial_layer_call_fn_337800305
(__inference_rial_layer_call_fn_337801956
(__inference_rial_layer_call_fn_337802013
(__inference_rial_layer_call_fn_337800718²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z/trace_0z0trace_1z1trace_2z2trace_3
´
3trace_0
4trace_1
5trace_2
6trace_32É
C__inference_rial_layer_call_and_return_conditional_losses_337802172
C__inference_rial_layer_call_and_return_conditional_losses_337802345
C__inference_rial_layer_call_and_return_conditional_losses_337800799
C__inference_rial_layer_call_and_return_conditional_losses_337800880²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z3trace_0z4trace_1z5trace_2z6trace_3
óBð
$__inference__wrapped_model_337799498input_1input_2input_3input_4input_5"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø
7layer_with_weights-0
7layer-0
8layer_with_weights-1
8layer-1
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_sequential
ø
?layer_with_weights-0
?layer-0
@layer_with_weights-1
@layer-1
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_sequential
µ
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
µ
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
¥
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
(
Y	keras_api"
_tf_keras_layer
è
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`_random_generator

 kernel
!recurrent_kernel
"bias"
_tf_keras_layer
è
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator

#kernel
$recurrent_kernel
%bias"
_tf_keras_layer
ø
hlayer_with_weights-0
hlayer-0
ilayer_with_weights-1
ilayer-1
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_sequential

piter
	qdecay
rlearning_rate
smomentum
trho
rms
rms
rms
rms
rms
rms 
rms¡
rms¢
rms£
rms¤
 rms¥
!rms¦
"rms§
#rms¨
$rms©
%rmsª
&rms«
'rms¬
(rms­
)rms®momentum¯momentum°momentum±momentum²momentum³momentum´momentumµmomentum¶momentum·momentum¸ momentum¹!momentumº"momentum»#momentum¼$momentum½%momentum¾&momentum¿'momentumÀ(momentumÁ)momentumÂ"
	optimizer
ë
utrace_0
vtrace_1
wtrace_2
xtrace_3
ytrace_4
ztrace_52Ì
__inference_call_337801045
__inference_call_337801204
__inference_call_337801363
__inference_call_337801522
__inference_call_337801681
__inference_call_337801840¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zutrace_0zvtrace_1zwtrace_2zxtrace_3zytrace_4zztrace_5
,
{serving_default"
signature_map
:@2dense/kernel
:@2
dense/bias
!:	@2dense_1/kernel
:2dense_1/bias
!:	2dense_2/kernel
:2dense_2/bias
(:&2batch_normalization/gamma
':%2batch_normalization/beta
0:. (2batch_normalization/moving_mean
4:2 (2#batch_normalization/moving_variance
':%	2embedding/embeddings
):'	2embedding_1/embeddings
#:!
¬2gru_cell/kernel
,:*	d¬2gru_cell/recurrent_kernel
 :	¬2gru_cell/bias
$:"	d¬2gru_cell_1/kernel
.:,	d¬2gru_cell_1/recurrent_kernel
": 	¬2gru_cell_1/bias
 :d@2dense_3/kernel
:@2dense_3/bias
 :@2dense_4/kernel
:2dense_4/bias
.
0
1"
trackable_list_wrapper
_
0
	1

2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
(__inference_rial_layer_call_fn_337800305input_1input_2input_3input_4input_5"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
(__inference_rial_layer_call_fn_337801956input/0input/1input/2input/3input/4"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
(__inference_rial_layer_call_fn_337802013input/0input/1input/2input/3input/4"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
(__inference_rial_layer_call_fn_337800718input_1input_2input_3input_4input_5"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
C__inference_rial_layer_call_and_return_conditional_losses_337802172input/0input/1input/2input/3input/4"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
C__inference_rial_layer_call_and_return_conditional_losses_337802345input/0input/1input/2input/3input/4"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
C__inference_rial_layer_call_and_return_conditional_losses_337800799input_1input_2input_3input_4input_5"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
C__inference_rial_layer_call_and_return_conditional_losses_337800880input_1input_2input_3input_4input_5"²
©²¥
FullArgSpec(
args 
jself
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
½
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
ö
trace_0
trace_1
trace_2
trace_32
.__inference_sequential_layer_call_fn_337799551
.__inference_sequential_layer_call_fn_337802358
.__inference_sequential_layer_call_fn_337802371
.__inference_sequential_layer_call_fn_337799624À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
â
trace_0
trace_1
trace_2
trace_32ï
I__inference_sequential_layer_call_and_return_conditional_losses_337802389
I__inference_sequential_layer_call_and_return_conditional_losses_337802407
I__inference_sequential_layer_call_and_return_conditional_losses_337799638
I__inference_sequential_layer_call_and_return_conditional_losses_337799652À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses
	¡axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
þ
§trace_0
¨trace_1
©trace_2
ªtrace_32
0__inference_sequential_1_layer_call_fn_337799783
0__inference_sequential_1_layer_call_fn_337802424
0__inference_sequential_1_layer_call_fn_337802441
0__inference_sequential_1_layer_call_fn_337799862À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z§trace_0z¨trace_1z©trace_2zªtrace_3
ê
«trace_0
¬trace_1
­trace_2
®trace_32÷
K__inference_sequential_1_layer_call_and_return_conditional_losses_337802468
K__inference_sequential_1_layer_call_and_return_conditional_losses_337802509
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799880
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799898À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z«trace_0z¬trace_1z­trace_2z®trace_3
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
ó
´trace_02Ô
-__inference_embedding_layer_call_fn_337802516¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z´trace_0

µtrace_02ï
H__inference_embedding_layer_call_and_return_conditional_losses_337802526¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zµtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
õ
»trace_02Ö
/__inference_embedding_1_layer_call_fn_337802533¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z»trace_0

¼trace_02ñ
J__inference_embedding_1_layer_call_and_return_conditional_losses_337802543¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¼trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
í
Âtrace_02Î
'__inference_add_layer_call_fn_337802551¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÂtrace_0

Ãtrace_02é
B__inference_add_layer_call_and_return_conditional_losses_337802561¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÃtrace_0
"
_generic_user_object
5
 0
!1
"2"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
Ø
Étrace_0
Êtrace_12
,__inference_gru_cell_layer_call_fn_337802575
,__inference_gru_cell_layer_call_fn_337802589¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÉtrace_0zÊtrace_1

Ëtrace_0
Ìtrace_12Ó
G__inference_gru_cell_layer_call_and_return_conditional_losses_337802628
G__inference_gru_cell_layer_call_and_return_conditional_losses_337802667¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zËtrace_0zÌtrace_1
"
_generic_user_object
5
#0
$1
%2"
trackable_list_wrapper
5
#0
$1
%2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Ü
Òtrace_0
Ótrace_12¡
.__inference_gru_cell_1_layer_call_fn_337802681
.__inference_gru_cell_1_layer_call_fn_337802695¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÒtrace_0zÓtrace_1

Ôtrace_0
Õtrace_12×
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337802734
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337802773¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÔtrace_0zÕtrace_1
"
_generic_user_object
Á
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
Á
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses

(kernel
)bias"
_tf_keras_layer
<
&0
'1
(2
)3"
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
þ
çtrace_0
ètrace_1
étrace_2
êtrace_32
0__inference_sequential_2_layer_call_fn_337799951
0__inference_sequential_2_layer_call_fn_337802786
0__inference_sequential_2_layer_call_fn_337802799
0__inference_sequential_2_layer_call_fn_337800024À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zçtrace_0zètrace_1zétrace_2zêtrace_3
ê
ëtrace_0
ìtrace_1
ítrace_2
îtrace_32÷
K__inference_sequential_2_layer_call_and_return_conditional_losses_337802817
K__inference_sequential_2_layer_call_and_return_conditional_losses_337802835
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800038
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800052À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zëtrace_0zìtrace_1zítrace_2zîtrace_3
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
òBï
__inference_call_337801045input/0input/1input/2input/3input/4"¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
òBï
__inference_call_337801204input/0input/1input/2input/3input/4"¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
òBï
__inference_call_337801363input/0input/1input/2input/3input/4"¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
òBï
__inference_call_337801522input/0input/1input/2input/3input/4"¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
òBï
__inference_call_337801681input/0input/1input/2input/3input/4"¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
òBï
__inference_call_337801840input/0input/1input/2input/3input/4"¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ðBí
'__inference_signature_wrapper_337801899input_1input_2input_3input_4input_5"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
ôtrace_02Ð
)__inference_dense_layer_call_fn_337802844¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zôtrace_0

õtrace_02ë
D__inference_dense_layer_call_and_return_conditional_losses_337802855¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zõtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ñ
ûtrace_02Ò
+__inference_dense_1_layer_call_fn_337802864¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zûtrace_0

ütrace_02í
F__inference_dense_1_layer_call_and_return_conditional_losses_337802875¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zütrace_0
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_layer_call_fn_337799551dense_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bý
.__inference_sequential_layer_call_fn_337802358inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bý
.__inference_sequential_layer_call_fn_337802371inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
.__inference_sequential_layer_call_fn_337799624dense_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
I__inference_sequential_layer_call_and_return_conditional_losses_337802389inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
I__inference_sequential_layer_call_and_return_conditional_losses_337802407inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 B
I__inference_sequential_layer_call_and_return_conditional_losses_337799638dense_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 B
I__inference_sequential_layer_call_and_return_conditional_losses_337799652dense_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_dense_2_layer_call_fn_337802884¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_dense_2_layer_call_and_return_conditional_losses_337802895¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
ä
trace_0
trace_12©
7__inference_batch_normalization_layer_call_fn_337802908
7__inference_batch_normalization_layer_call_fn_337802921´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12ß
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337802941
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337802975´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
0__inference_sequential_1_layer_call_fn_337799783dense_2_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bÿ
0__inference_sequential_1_layer_call_fn_337802424inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bÿ
0__inference_sequential_1_layer_call_fn_337802441inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
0__inference_sequential_1_layer_call_fn_337799862dense_2_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
K__inference_sequential_1_layer_call_and_return_conditional_losses_337802468inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
K__inference_sequential_1_layer_call_and_return_conditional_losses_337802509inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤B¡
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799880dense_2_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤B¡
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799898dense_2_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
áBÞ
-__inference_embedding_layer_call_fn_337802516inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
H__inference_embedding_layer_call_and_return_conditional_losses_337802526inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ãBà
/__inference_embedding_1_layer_call_fn_337802533inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
J__inference_embedding_1_layer_call_and_return_conditional_losses_337802543inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
'__inference_add_layer_call_fn_337802551inputs/0inputs/1inputs/2inputs/3"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
B__inference_add_layer_call_and_return_conditional_losses_337802561inputs/0inputs/1inputs/2inputs/3"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_gru_cell_layer_call_fn_337802575inputsstates"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
,__inference_gru_cell_layer_call_fn_337802589inputsstates"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_gru_cell_layer_call_and_return_conditional_losses_337802628inputsstates"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_gru_cell_layer_call_and_return_conditional_losses_337802667inputsstates"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_gru_cell_1_layer_call_fn_337802681inputsstates"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
.__inference_gru_cell_1_layer_call_fn_337802695inputsstates"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¡B
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337802734inputsstates"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¡B
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337802773inputsstates"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_dense_3_layer_call_fn_337802984¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_dense_3_layer_call_and_return_conditional_losses_337802995¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_dense_4_layer_call_fn_337803004¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_dense_4_layer_call_and_return_conditional_losses_337803015¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
0__inference_sequential_2_layer_call_fn_337799951dense_3_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bÿ
0__inference_sequential_2_layer_call_fn_337802786inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bÿ
0__inference_sequential_2_layer_call_fn_337802799inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
0__inference_sequential_2_layer_call_fn_337800024dense_3_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
K__inference_sequential_2_layer_call_and_return_conditional_losses_337802817inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
K__inference_sequential_2_layer_call_and_return_conditional_losses_337802835inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤B¡
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800038dense_3_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤B¡
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800052dense_3_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_layer_call_fn_337802844inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_dense_layer_call_and_return_conditional_losses_337802855inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_1_layer_call_fn_337802864inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_1_layer_call_and_return_conditional_losses_337802875inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_2_layer_call_fn_337802884inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_2_layer_call_and_return_conditional_losses_337802895inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
7__inference_batch_normalization_layer_call_fn_337802908inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ýBú
7__inference_batch_normalization_layer_call_fn_337802921inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337802941inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337802975inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_3_layer_call_fn_337802984inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_3_layer_call_and_return_conditional_losses_337802995inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_4_layer_call_fn_337803004inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_4_layer_call_and_return_conditional_losses_337803015inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(:&@2RMSprop/dense/kernel/rms
": @2RMSprop/dense/bias/rms
+:)	@2RMSprop/dense_1/kernel/rms
%:#2RMSprop/dense_1/bias/rms
+:)	2RMSprop/dense_2/kernel/rms
%:#2RMSprop/dense_2/bias/rms
2:02%RMSprop/batch_normalization/gamma/rms
1:/2$RMSprop/batch_normalization/beta/rms
1:/	2 RMSprop/embedding/embeddings/rms
3:1	2"RMSprop/embedding_1/embeddings/rms
-:+
¬2RMSprop/gru_cell/kernel/rms
6:4	d¬2%RMSprop/gru_cell/recurrent_kernel/rms
*:(	¬2RMSprop/gru_cell/bias/rms
.:,	d¬2RMSprop/gru_cell_1/kernel/rms
8:6	d¬2'RMSprop/gru_cell_1/recurrent_kernel/rms
,:*	¬2RMSprop/gru_cell_1/bias/rms
*:(d@2RMSprop/dense_3/kernel/rms
$:"@2RMSprop/dense_3/bias/rms
*:(@2RMSprop/dense_4/kernel/rms
$:"2RMSprop/dense_4/bias/rms
-:+@2RMSprop/dense/kernel/momentum
':%@2RMSprop/dense/bias/momentum
0:.	@2RMSprop/dense_1/kernel/momentum
*:(2RMSprop/dense_1/bias/momentum
0:.	2RMSprop/dense_2/kernel/momentum
*:(2RMSprop/dense_2/bias/momentum
7:52*RMSprop/batch_normalization/gamma/momentum
6:42)RMSprop/batch_normalization/beta/momentum
6:4	2%RMSprop/embedding/embeddings/momentum
8:6	2'RMSprop/embedding_1/embeddings/momentum
2:0
¬2 RMSprop/gru_cell/kernel/momentum
;:9	d¬2*RMSprop/gru_cell/recurrent_kernel/momentum
/:-	¬2RMSprop/gru_cell/bias/momentum
3:1	d¬2"RMSprop/gru_cell_1/kernel/momentum
=:;	d¬2,RMSprop/gru_cell_1/recurrent_kernel/momentum
1:/	¬2 RMSprop/gru_cell_1/bias/momentum
/:-d@2RMSprop/dense_3/kernel/momentum
):'@2RMSprop/dense_3/bias/momentum
/:-@2RMSprop/dense_4/kernel/momentum
):'2RMSprop/dense_4/bias/momentum¥
$__inference__wrapped_model_337799498ü" !%#$&'()Ê¢Æ
¾¢º
·¢³
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
!
input_4ÿÿÿÿÿÿÿÿÿ
%"
input_5ÿÿÿÿÿÿÿÿÿd
ª "ª
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿd
.
output_3"
output_3ÿÿÿÿÿÿÿÿÿd
B__inference_add_layer_call_and_return_conditional_losses_337802561Ö«¢§
¢

# 
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
# 
inputs/3ÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 õ
'__inference_add_layer_call_fn_337802551É«¢§
¢

# 
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
# 
inputs/3ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿº
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337802941d4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 º
R__inference_batch_normalization_layer_call_and_return_conditional_losses_337802975d4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_layer_call_fn_337802908W4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_layer_call_fn_337802921W4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
__inference_call_337801045ù" !%#$&'()¢
¢
¢

input/0

input/1

input/2

input/3

input/4d
ª "?¢<

0

1d

2d
__inference_call_337801204ù" !%#$&'()¢
¢
¢

input/0

input/1	

input/2	

input/3

input/4d
ª "?¢<

0

1d

2d
__inference_call_337801363ù" !%#$&'()¢
¢
¢

input/0

input/1

input/2	

input/3

input/4d
ª "?¢<

0

1d

2d
__inference_call_337801522ù" !%#$&'()¢
¢
¢

input/0

input/1	

input/2

input/3

input/4d
ª "?¢<

0

1d

2d
__inference_call_337801681ù" !%#$&'()¢
¢
¢

input/0 

input/1 	

input/2 	

input/3 

input/4 d
ª "?¢<

0 

1 d

2 dà
__inference_call_337801840Á" !%#$&'()Ê¢Æ
¾¢º
·¢³
!
input/0ÿÿÿÿÿÿÿÿÿ
!
input/1ÿÿÿÿÿÿÿÿÿ
!
input/2ÿÿÿÿÿÿÿÿÿ
!
input/3ÿÿÿÿÿÿÿÿÿ
%"
input/4ÿÿÿÿÿÿÿÿÿd
ª "Z¢W

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿd

2ÿÿÿÿÿÿÿÿÿd§
F__inference_dense_1_layer_call_and_return_conditional_losses_337802875]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_1_layer_call_fn_337802864P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_dense_2_layer_call_and_return_conditional_losses_337802895]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_2_layer_call_fn_337802884P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_3_layer_call_and_return_conditional_losses_337802995\&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dense_3_layer_call_fn_337802984O&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dense_4_layer_call_and_return_conditional_losses_337803015\()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_4_layer_call_fn_337803004O()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_layer_call_and_return_conditional_losses_337802855\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dense_layer_call_fn_337802844O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@®
J__inference_embedding_1_layer_call_and_return_conditional_losses_337802543`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
/__inference_embedding_1_layer_call_fn_337802533S/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
H__inference_embedding_layer_call_and_return_conditional_losses_337802526`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_embedding_layer_call_fn_337802516S/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ÷
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337802734©%#$U¢R
K¢H
 
inputsÿÿÿÿÿÿÿÿÿd
 
statesÿÿÿÿÿÿÿÿÿd
p 
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿd

0/1ÿÿÿÿÿÿÿÿÿd
 ÷
I__inference_gru_cell_1_layer_call_and_return_conditional_losses_337802773©%#$U¢R
K¢H
 
inputsÿÿÿÿÿÿÿÿÿd
 
statesÿÿÿÿÿÿÿÿÿd
p
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿd

0/1ÿÿÿÿÿÿÿÿÿd
 Î
.__inference_gru_cell_1_layer_call_fn_337802681%#$U¢R
K¢H
 
inputsÿÿÿÿÿÿÿÿÿd
 
statesÿÿÿÿÿÿÿÿÿd
p 
ª "=¢:

0ÿÿÿÿÿÿÿÿÿd

1ÿÿÿÿÿÿÿÿÿdÎ
.__inference_gru_cell_1_layer_call_fn_337802695%#$U¢R
K¢H
 
inputsÿÿÿÿÿÿÿÿÿd
 
statesÿÿÿÿÿÿÿÿÿd
p
ª "=¢:

0ÿÿÿÿÿÿÿÿÿd

1ÿÿÿÿÿÿÿÿÿdö
G__inference_gru_cell_layer_call_and_return_conditional_losses_337802628ª" !V¢S
L¢I
!
inputsÿÿÿÿÿÿÿÿÿ
 
statesÿÿÿÿÿÿÿÿÿd
p 
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿd

0/1ÿÿÿÿÿÿÿÿÿd
 ö
G__inference_gru_cell_layer_call_and_return_conditional_losses_337802667ª" !V¢S
L¢I
!
inputsÿÿÿÿÿÿÿÿÿ
 
statesÿÿÿÿÿÿÿÿÿd
p
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿd

0/1ÿÿÿÿÿÿÿÿÿd
 Í
,__inference_gru_cell_layer_call_fn_337802575" !V¢S
L¢I
!
inputsÿÿÿÿÿÿÿÿÿ
 
statesÿÿÿÿÿÿÿÿÿd
p 
ª "=¢:

0ÿÿÿÿÿÿÿÿÿd

1ÿÿÿÿÿÿÿÿÿdÍ
,__inference_gru_cell_layer_call_fn_337802589" !V¢S
L¢I
!
inputsÿÿÿÿÿÿÿÿÿ
 
statesÿÿÿÿÿÿÿÿÿd
p
ª "=¢:

0ÿÿÿÿÿÿÿÿÿd

1ÿÿÿÿÿÿÿÿÿd
C__inference_rial_layer_call_and_return_conditional_losses_337800799Õ" !%#$&'()Î¢Ê
Â¢¾
·¢³
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
!
input_4ÿÿÿÿÿÿÿÿÿ
%"
input_5ÿÿÿÿÿÿÿÿÿd
p 
ª "j¢g
`¢]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿd

0/2ÿÿÿÿÿÿÿÿÿd
 
C__inference_rial_layer_call_and_return_conditional_losses_337800880Õ" !%#$&'()Î¢Ê
Â¢¾
·¢³
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
!
input_4ÿÿÿÿÿÿÿÿÿ
%"
input_5ÿÿÿÿÿÿÿÿÿd
p
ª "j¢g
`¢]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿd

0/2ÿÿÿÿÿÿÿÿÿd
 
C__inference_rial_layer_call_and_return_conditional_losses_337802172Õ" !%#$&'()Î¢Ê
Â¢¾
·¢³
!
input/0ÿÿÿÿÿÿÿÿÿ
!
input/1ÿÿÿÿÿÿÿÿÿ
!
input/2ÿÿÿÿÿÿÿÿÿ
!
input/3ÿÿÿÿÿÿÿÿÿ
%"
input/4ÿÿÿÿÿÿÿÿÿd
p 
ª "j¢g
`¢]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿd

0/2ÿÿÿÿÿÿÿÿÿd
 
C__inference_rial_layer_call_and_return_conditional_losses_337802345Õ" !%#$&'()Î¢Ê
Â¢¾
·¢³
!
input/0ÿÿÿÿÿÿÿÿÿ
!
input/1ÿÿÿÿÿÿÿÿÿ
!
input/2ÿÿÿÿÿÿÿÿÿ
!
input/3ÿÿÿÿÿÿÿÿÿ
%"
input/4ÿÿÿÿÿÿÿÿÿd
p
ª "j¢g
`¢]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿd

0/2ÿÿÿÿÿÿÿÿÿd
 ò
(__inference_rial_layer_call_fn_337800305Å" !%#$&'()Î¢Ê
Â¢¾
·¢³
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
!
input_4ÿÿÿÿÿÿÿÿÿ
%"
input_5ÿÿÿÿÿÿÿÿÿd
p 
ª "Z¢W

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿd

2ÿÿÿÿÿÿÿÿÿdò
(__inference_rial_layer_call_fn_337800718Å" !%#$&'()Î¢Ê
Â¢¾
·¢³
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
!
input_4ÿÿÿÿÿÿÿÿÿ
%"
input_5ÿÿÿÿÿÿÿÿÿd
p
ª "Z¢W

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿd

2ÿÿÿÿÿÿÿÿÿdò
(__inference_rial_layer_call_fn_337801956Å" !%#$&'()Î¢Ê
Â¢¾
·¢³
!
input/0ÿÿÿÿÿÿÿÿÿ
!
input/1ÿÿÿÿÿÿÿÿÿ
!
input/2ÿÿÿÿÿÿÿÿÿ
!
input/3ÿÿÿÿÿÿÿÿÿ
%"
input/4ÿÿÿÿÿÿÿÿÿd
p 
ª "Z¢W

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿd

2ÿÿÿÿÿÿÿÿÿdò
(__inference_rial_layer_call_fn_337802013Å" !%#$&'()Î¢Ê
Â¢¾
·¢³
!
input/0ÿÿÿÿÿÿÿÿÿ
!
input/1ÿÿÿÿÿÿÿÿÿ
!
input/2ÿÿÿÿÿÿÿÿÿ
!
input/3ÿÿÿÿÿÿÿÿÿ
%"
input/4ÿÿÿÿÿÿÿÿÿd
p
ª "Z¢W

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿd

2ÿÿÿÿÿÿÿÿÿd¿
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799880p>¢;
4¢1
'$
dense_2_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¿
K__inference_sequential_1_layer_call_and_return_conditional_losses_337799898p>¢;
4¢1
'$
dense_2_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¸
K__inference_sequential_1_layer_call_and_return_conditional_losses_337802468i7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¸
K__inference_sequential_1_layer_call_and_return_conditional_losses_337802509i7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_sequential_1_layer_call_fn_337799783c>¢;
4¢1
'$
dense_2_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_1_layer_call_fn_337799862c>¢;
4¢1
'$
dense_2_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_1_layer_call_fn_337802424\7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_1_layer_call_fn_337802441\7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¼
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800038m&'()>¢;
4¢1
'$
dense_3_inputÿÿÿÿÿÿÿÿÿd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
K__inference_sequential_2_layer_call_and_return_conditional_losses_337800052m&'()>¢;
4¢1
'$
dense_3_inputÿÿÿÿÿÿÿÿÿd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
K__inference_sequential_2_layer_call_and_return_conditional_losses_337802817f&'()7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
K__inference_sequential_2_layer_call_and_return_conditional_losses_337802835f&'()7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_sequential_2_layer_call_fn_337799951`&'()>¢;
4¢1
'$
dense_3_inputÿÿÿÿÿÿÿÿÿd
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_2_layer_call_fn_337800024`&'()>¢;
4¢1
'$
dense_3_inputÿÿÿÿÿÿÿÿÿd
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_2_layer_call_fn_337802786Y&'()7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_2_layer_call_fn_337802799Y&'()7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "ÿÿÿÿÿÿÿÿÿ¹
I__inference_sequential_layer_call_and_return_conditional_losses_337799638l<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¹
I__inference_sequential_layer_call_and_return_conditional_losses_337799652l<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ´
I__inference_sequential_layer_call_and_return_conditional_losses_337802389g7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ´
I__inference_sequential_layer_call_and_return_conditional_losses_337802407g7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_sequential_layer_call_fn_337799551_<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_layer_call_fn_337799624_<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_layer_call_fn_337802358Z7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_layer_call_fn_337802371Z7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿØ
'__inference_signature_wrapper_337801899¬" !%#$&'()ú¢ö
¢ 
îªê
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
,
input_2!
input_2ÿÿÿÿÿÿÿÿÿ
,
input_3!
input_3ÿÿÿÿÿÿÿÿÿ
,
input_4!
input_4ÿÿÿÿÿÿÿÿÿ
0
input_5%"
input_5ÿÿÿÿÿÿÿÿÿd"ª
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿd
.
output_3"
output_3ÿÿÿÿÿÿÿÿÿd