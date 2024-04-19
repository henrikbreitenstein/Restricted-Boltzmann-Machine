def construct_ising(N, M, J, L):
    
    J_x = np.array([[0, 1], [1, 0]])
    J_z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    eyes = [I]
    for i in range(1, N*M):
        eyes.append(np.kron(eyes[i-1], I))
    right_list = []
    up_list = []
    single_list = []
    for i in range(M):
        for j in range(N):
            
            if (j < (N-2)) and (j > 0):
                before = N*i+j
                after = N*M-1 - before - 2
                mat_list_right = [eyes[before-1], J*J_z, J*J_z, eyes[after]]
                mat_list_single = [eyes[before-1], L*J_x, eyes[after+1]]   
            else:
                mat_list_right = [I]*N*M
                mat_list_single = [I]*N*M    
                mat_list_right[N*i+j] = J*J_z
                mat_list_right[N*i+(j+1)%N] = J*J_z
                mat_list_single[N*i+j] = L*J_x
            
            if (i < (M-1)) and (i > 0):
                before = N*i+j
                after = N*(M-1) - 1 - before - N
                mat_list_up = [
                    eyes[before-1], 
                    J*J_z, 
                    eyes[N-1], 
                    J*J_z, 
                    eyes[after]
                ]
            else:
                mat_list_up = [I]*N*M
                mat_list_up[N*i+j] = J*J_z
                mat_list_up[N*((i+1)%M)+j] = J*J_z
           
            right = mat_list_right[0]
            up = mat_list_up[0]
            single = mat_list_single[0]

            for mat_right in mat_list_right[1:]:
                right = np.kron(mat_right, right)
            for mat_up in mat_list_up[1:]:
                up = np.kron(mat_up, up)
            for mat_single in mat_list_single[1:]:
                single = np.kron(mat_single, single)
            
            right_list.append(right)
            up_list.append(up)
            single_list.append(single)
    
    return right_list, up_list, single_list

def ising_true(N, M, J, L): 
    right, up, single = construct_ising(N, 1, J, L)
    H = np.zeros([2**(N*M), 2**(N*M)])
    for mat in right:
        H += mat
    for mat in up:
        H += mat
    for mat in single:
        H += mat
    if torch.cuda.is_available():
        H = torch.tensor(H, device=torch.device('cuda'))
    eig_vals = torch.linalg.eigvalsh(H)
    return min(eig_vals).item()

def ising_pairs(basis, N, M):
    basis, diff_basis = basis
    pairs = torch.zeros_like(diff_basis)
    size = basis.shape[0]
    for i in range(size):
        for j in range(size):
            xor = torch.logical_xor(basis[i], basis[j])
            if torch.sum(xor) == 2:
                wxor = torch.where(xor)[0]
                diff = abs(wxor[0] - wxor[1])
                if wxor[0]%N < wxor[1]%N:
                    if (diff == 1) or (diff==(N-1)):
                        pairs[i, j] = 1
                if wxor[0]%N > wxor[1]%N:
                    if (diff == N) or (diff==(N*(M-1))):
                        pairs[i, j] = 1 
    return basis, diff_basis, pairs

# N_0 = torch.sum(basis == 0, dim=-1)
# N_1 = torch.sum(basis == 1, dim=-1)
#  
# H_0 = 0.5*eps*(N_1-N_0)
# H_eps = torch.zeros_like(H_0)
# H_eps[non_zero_mask] = H_0[non_zero_mask]


# H_V = 0.5*V*torch.sum(non_zero[:, None]*(non_zero_diff == 2), dim=0)/weight
# H_W = 0.5*W*torch.sum(non_zero[:, None]*(non_zero_diff == 1), dim=0)/weight
#
def ising_2d_coupling(N, M, J, samples):
    
    pm = 2*samples - 1
    pm = torch.reshape(pm, (pm.shape[0], M, N))
    up = torch.roll(pm, 1, 1)
    right = torch.roll(pm, 1, 2)
    H_J = up*pm + right*pm
    H_J = torch.sum(H_J, dim=-1)
    H_J = torch.sum(H_J, dim=-1)
    return H_J

def ising_basis_set(N, M, J, L, basis_set):
    basis, diff_basis = basis_set
    if M == 1:
        H_J = ising_1d_coupling(J, basis)
    else:
        H_J = ising_2d_coupling(N, M, J, basis)
    return basis, diff_basis, H_J

def ising_local(J, L, samples, basis):
    
    basis, diff_basis, H_J = basis
    weight, non_zero_mask, mask = amplitudes(samples, basis)
    
    non_zero = weight[non_zero_mask]
    non_zero_diff = diff_basis[non_zero_mask]

    H_L = L*torch.sum(non_zero[:, None]*(non_zero_diff==1), dim=0)/weight
    H_s = torch.zeros_like(H_J)
    H_s[non_zero_mask] = H_J[non_zero_mask]
    
    H = H_s + H_L

    return H[mask], torch.var(H), H, weight

def ising_1d_coupling(J, samples):
    pm = 2*samples - 1
    shift_r = torch.roll(pm, 1, 1)
    H_J = J*torch.sum(pm*shift_r, dim=-1)
    return H_J

# dPsidvb = (basis - vb)
# dPsidhb = 1/(torch.exp(-hb-basis@W)+ 1)
# dPsidW  = basis[:, :, None]*dPsidhb[:, None, :]
# basis_mean = torch.mean(E)
# E_diff = weight*(E - basis_mean)
# E_mean = torch.mean(E_local)
