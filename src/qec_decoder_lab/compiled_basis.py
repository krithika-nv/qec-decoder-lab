"""Optional compiled GF(2) elimination for BP-guided syndrome recovery.

A fault representation is stored as a packed uint64 vector so every XOR uses
native machine words, rather than Python's per-fault integer/dict iteration.
The pivot convention and ordering exactly match `recovery._basis`.
"""
from __future__ import annotations

import numpy as np
try:
    import numba
except ImportError as exc:
    raise ImportError('compiled basis requires compatible numba: pip install numba') from exc


def pack_detector_masks(det_masks: list[int], detectors: int) -> np.ndarray:
    if detectors < 0:
        raise ValueError('negative detector count')
    nwords = max(1, (detectors+63)//64)
    result = np.zeros((len(det_masks),nwords), dtype=np.uint64)
    max_mask=1<<detectors
    for j,mask in enumerate(det_masks):
        if mask<0 or mask>=max_mask:
            raise ValueError('fault detector mask outside detector range')
        for w in range(nwords):
            result[j,w]=mask & ((1<<64)-1)
            mask >>= 64
    return result


@numba.njit(cache=True)
def _run_basis(mask_words: np.ndarray, order: np.ndarray, detectors: int,
               target_rank: int):
    nfaults,nwords = mask_words.shape
    nrep=(nfaults+63)//64
    exists=np.zeros(detectors, dtype=np.uint8)
    pivots=np.zeros((detectors,nwords), dtype=np.uint64)
    representations=np.zeros((detectors,nrep), dtype=np.uint64)
    current = np.empty(nwords, dtype=np.uint64)
    represent = np.zeros(nrep, dtype=np.uint64)
    rank=0
    for index in range(len(order)):
        j=int(order[index])
        if j<0 or j>=nfaults:
            raise ValueError('invalid fault ordering')
        for w in range(nwords):
            current[w]=mask_words[j,w]
        represent[:]=np.uint64(0)
        represent[j//64]=np.uint64(1)<<np.uint64(j%64)
        while True:
            pivot=-1
            for w in range(nwords):
                x=current[w]
                if x != 0:
                    bit=0
                    while (x & np.uint64(1)) == 0:
                        x=x>>np.uint64(1)
                        bit+=1
                    pivot=w*64+bit
                    break
            if pivot<0:
                break
            if exists[pivot]==0:
                exists[pivot]=1
                rank+=1
                for w in range(nwords): pivots[pivot,w]=current[w]
                for w in range(nrep): representations[pivot,w]=represent[w]
                break
            for w in range(nwords):
                current[w]^=pivots[pivot,w]
            for w in range(nrep):
                represent[w]^=representations[pivot,w]
        if target_rank>0 and rank==target_rank:
            break
    return exists,pivots,representations


def _combine_words(row: np.ndarray) -> int:
    x=0
    for w in range(len(row)-1,-1,-1):
        x=(x<<64) | int(row[w])
    return x


def packed_basis(mask_words: np.ndarray, order: np.ndarray,
                 detectors: int, target_rank: int) -> dict[int,tuple[int,int]]:
    if target_rank<0 or target_rank>detectors:
        raise ValueError('invalid target rank')
    if len(order)!=len(mask_words) or len(set(map(int,order)))!=len(order):
        raise ValueError('fault order must be a permutation')
    exists,columns,reps=_run_basis(mask_words,np.asarray(order,dtype=np.int64),
                                    detectors, target_rank)
    return {i:(_combine_words(columns[i]),_combine_words(reps[i]))
            for i in np.flatnonzero(exists)}
