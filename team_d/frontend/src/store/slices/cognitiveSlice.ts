/**
 * 认知状态管理 Slice
 * 处理认知状态数据、实时更新、历史记录等
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { CognitiveState, CognitiveStateHistory, CognitiveConcept } from '@/types';
import * as cognitiveService from '@/services/cognitiveService';

// 状态类型定义
interface CognitiveSliceState {
  // 当前认知状态
  currentState: CognitiveState | null;
  
  // 历史记录
  history: CognitiveState[];
  
  // 历史统计数据
  statistics: CognitiveStateHistory | null;
  
  // 实时数据流
  realtimeEnabled: boolean;
  
  // 加载状态
  loading: {
    current: boolean;
    history: boolean;
    analysis: boolean;
  };
  
  // 错误状态
  error: string | null;
  
  // 数据更新时间
  lastUpdated: number | null;
  
  // 预测准确性
  confidence: number;
  
  // 概念间关系权重
  conceptRelationships: Record<string, Record<string, number>> | null;
}

// 初始状态
const initialState: CognitiveSliceState = {
  currentState: null,
  history: [],
  statistics: null,
  realtimeEnabled: false,
  loading: {
    current: false,
    history: false,
    analysis: false,
  },
  error: null,
  lastUpdated: null,
  confidence: 0,
  conceptRelationships: null,
};

// ============ 异步 Actions ============

/** 获取最新认知状态 */
export const fetchLatestCognitiveStateAsync = createAsyncThunk(
  'cognitive/fetchLatest',
  async (userId: string) => {
    const response = await cognitiveService.getLatestState(userId);
    return response;
  }
);

/** 获取认知状态历史 */
export const fetchCognitiveHistoryAsync = createAsyncThunk(
  'cognitive/fetchHistory',
  async (params: {
    userId: string;
    timeRange?: {
      start: number;
      end: number;
    };
    limit?: number;
  }) => {
    const response = await cognitiveService.getHistory(params);
    return response;
  }
);

/** 分析认知状态趋势 */
export const analyzeCognitiveStateAsync = createAsyncThunk(
  'cognitive/analyzeState',
  async (params: {
    userId: string;
    timeRange: {
      start: number;
      end: number;
    };
    analysisType: 'trend' | 'correlation' | 'anomaly';
  }) => {
    const response = await cognitiveService.analyzeState(params);
    return response;
  }
);

/** 预测认知状态变化 */
export const predictCognitiveStateAsync = createAsyncThunk(
  'cognitive/predictState',
  async (params: {
    userId: string;
    currentState: CognitiveState;
    contextData?: any;
    timeHorizon?: number; // 预测时长（分钟）
  }) => {
    const response = await cognitiveService.predictState(params);
    return response;
  }
);

/** 导出认知状态数据 */
export const exportCognitiveDataAsync = createAsyncThunk(
  'cognitive/exportData',
  async (params: {
    userId: string;
    format: 'csv' | 'json' | 'excel';
    timeRange?: {
      start: number;
      end: number;
    };
    includeMetadata?: boolean;
  }) => {
    const response = await cognitiveService.exportData(params);
    return response;
  }
);

// ============ Slice 定义 ============

const cognitiveSlice = createSlice({
  name: 'cognitive',
  initialState,
  reducers: {
    // 清除错误状态
    clearError: (state) => {
      state.error = null;
    },
    
    // 实时更新认知状态（通过WebSocket）
    updateCurrentState: (state, action: PayloadAction<CognitiveState>) => {
      const newState = action.payload;
      
      // 更新当前状态
      state.currentState = newState;
      state.lastUpdated = Date.now();
      state.confidence = newState.confidence;
      
      // 添加到历史记录
      state.history.unshift(newState);
      
      // 保持历史记录数量限制（最多1000条）
      if (state.history.length > 1000) {
        state.history = state.history.slice(0, 1000);
      }
      
      // 更新统计信息
      updateStatistics(state);
    },
    
    // 批量添加历史状态
    addHistoryStates: (state, action: PayloadAction<CognitiveState[]>) => {
      const newStates = action.payload;
      
      // 按时间戳排序并去重
      const allStates = [...state.history, ...newStates];
      const uniqueStates = allStates.filter((state, index, arr) => 
        arr.findIndex(s => s.timestamp === state.timestamp) === index
      );
      
      // 按时间戳降序排序
      state.history = uniqueStates.sort((a, b) => b.timestamp - a.timestamp);
      
      // 限制数量
      if (state.history.length > 1000) {
        state.history = state.history.slice(0, 1000);
      }
      
      updateStatistics(state);
    },
    
    // 启用/禁用实时更新
    setRealtimeEnabled: (state, action: PayloadAction<boolean>) => {
      state.realtimeEnabled = action.payload;
    },
    
    // 更新概念间关系权重
    updateConceptRelationships: (state, action: PayloadAction<Record<string, Record<string, number>>>) => {
      state.conceptRelationships = action.payload;
    },
    
    // 设置特定加载状态
    setLoadingState: (state, action: PayloadAction<{ type: keyof typeof initialState.loading; loading: boolean }>) => {
      const { type, loading } = action.payload;
      state.loading[type] = loading;
    },
    
    // 清空历史数据
    clearHistory: (state) => {
      state.history = [];
      state.statistics = null;
    },
    
    // 重置所有状态
    resetState: (state) => {
      Object.assign(state, initialState);
    },
  },
  extraReducers: (builder) => {
    // ====== 获取最新认知状态 ======
    builder
      .addCase(fetchLatestCognitiveStateAsync.pending, (state) => {
        state.loading.current = true;
        state.error = null;
      })
      .addCase(fetchLatestCognitiveStateAsync.fulfilled, (state, action) => {
        state.loading.current = false;
        state.currentState = action.payload;
        state.lastUpdated = Date.now();
        state.confidence = action.payload.confidence;
        
        // 更新历史记录
        const existingIndex = state.history.findIndex(
          s => s.timestamp === action.payload.timestamp
        );
        if (existingIndex === -1) {
          state.history.unshift(action.payload);
        }
        
        updateStatistics(state);
      })
      .addCase(fetchLatestCognitiveStateAsync.rejected, (state, action) => {
        state.loading.current = false;
        state.error = action.error.message || '获取认知状态失败';
      });
    
    // ====== 获取历史记录 ======
    builder
      .addCase(fetchCognitiveHistoryAsync.pending, (state) => {
        state.loading.history = true;
        state.error = null;
      })
      .addCase(fetchCognitiveHistoryAsync.fulfilled, (state, action) => {
        state.loading.history = false;
        
        const newHistory = action.payload.states;
        const allStates = [...state.history, ...newHistory];
        
        // 去重并排序
        const uniqueStates = allStates.filter((state, index, arr) => 
          arr.findIndex(s => s.timestamp === state.timestamp) === index
        );
        
        state.history = uniqueStates.sort((a, b) => b.timestamp - a.timestamp);
        
        // 更新统计信息
        if (action.payload.statistics) {
          state.statistics = action.payload;
        }
        
        updateStatistics(state);
      })
      .addCase(fetchCognitiveHistoryAsync.rejected, (state, action) => {
        state.loading.history = false;
        state.error = action.error.message || '获取历史记录失败';
      });
    
    // ====== 分析认知状态 ======
    builder
      .addCase(analyzeCognitiveStateAsync.pending, (state) => {
        state.loading.analysis = true;
        state.error = null;
      })
      .addCase(analyzeCognitiveStateAsync.fulfilled, (state, action) => {
        state.loading.analysis = false;
        
        // 更新分析结果
        if (action.payload.conceptRelationships) {
          state.conceptRelationships = action.payload.conceptRelationships;
        }
        
        if (action.payload.statistics) {
          state.statistics = {
            ...state.statistics,
            ...action.payload.statistics,
          };
        }
      })
      .addCase(analyzeCognitiveStateAsync.rejected, (state, action) => {
        state.loading.analysis = false;
        state.error = action.error.message || '认知状态分析失败';
      });
    
    // ====== 预测认知状态 ======
    builder
      .addCase(predictCognitiveStateAsync.pending, (state) => {
        state.loading.analysis = true;
      })
      .addCase(predictCognitiveStateAsync.fulfilled, (state, action) => {
        state.loading.analysis = false;
        
        // 将预测结果添加到历史记录中，标记为预测数据
        const predictions = action.payload.predictions;
        if (predictions && predictions.length > 0) {
          const predictedStates = predictions.map((pred: any) => ({
            ...pred,
            metadata: {
              ...pred.metadata,
              isPrediction: true,
            },
          }));
          
          state.history = [...predictedStates, ...state.history];
        }
      })
      .addCase(predictCognitiveStateAsync.rejected, (state, action) => {
        state.loading.analysis = false;
        state.error = action.error.message || '认知状态预测失败';
      });
  },
});

// ============ 辅助函数 ============

/** 更新统计信息 */
function updateStatistics(state: CognitiveSliceState) {
  if (state.history.length === 0) return;
  
  const concepts: CognitiveConcept[] = [
    'attention', 'memory', 'comprehension', 'creativity',
    'motivation', 'emotion', 'confidence', 'fatigue'
  ];
  
  // 计算平均值
  const averageValues: Record<CognitiveConcept, number> = {} as any;
  concepts.forEach(concept => {
    const values = state.history.map(s => s.values[concept]).filter(v => v !== undefined);
    averageValues[concept] = values.length > 0 ? 
      values.reduce((sum, val) => sum + val, 0) / values.length : 0;
  });
  
  // 计算趋势
  const trends: Record<CognitiveConcept, 'increasing' | 'decreasing' | 'stable'> = {} as any;
  concepts.forEach(concept => {
    const recentStates = state.history.slice(0, 10); // 最近10个状态
    if (recentStates.length < 3) {
      trends[concept] = 'stable';
      return;
    }
    
    const recentValues = recentStates.map(s => s.values[concept]);
    const firstHalf = recentValues.slice(0, Math.floor(recentValues.length / 2));
    const secondHalf = recentValues.slice(Math.floor(recentValues.length / 2));
    
    const firstAvg = firstHalf.reduce((sum, val) => sum + val, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((sum, val) => sum + val, 0) / secondHalf.length;
    
    const threshold = 0.05; // 5% 变化阈值
    if (secondAvg - firstAvg > threshold) {
      trends[concept] = 'increasing';
    } else if (firstAvg - secondAvg > threshold) {
      trends[concept] = 'decreasing';
    } else {
      trends[concept] = 'stable';
    }
  });
  
  // 更新统计信息
  if (!state.statistics) {
    state.statistics = {
      states: [],
      timeRange: { start: 0, end: 0 },
      statistics: { averageValues, trends },
    };
  } else {
    state.statistics.statistics = { averageValues, trends };
  }
  
  // 更新时间范围
  if (state.history.length > 0) {
    const timestamps = state.history.map(s => s.timestamp);
    state.statistics.timeRange = {
      start: Math.min(...timestamps),
      end: Math.max(...timestamps),
    };
  }
}

// ============ 导出 Actions 和 Selectors ============

export const {
  clearError,
  updateCurrentState,
  addHistoryStates,
  setRealtimeEnabled,
  updateConceptRelationships,
  setLoadingState,
  clearHistory,
  resetState,
} = cognitiveSlice.actions;

// Selectors
export const selectCognitive = (state: { cognitive: CognitiveSliceState }) => state.cognitive;
export const selectCurrentCognitiveState = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.currentState;
export const selectCognitiveHistory = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.history;
export const selectCognitiveStatistics = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.statistics;
export const selectCognitiveLoading = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.loading;
export const selectCognitiveError = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.error;
export const selectRealtimeEnabled = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.realtimeEnabled;
export const selectCognitiveConfidence = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.confidence;
export const selectConceptRelationships = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.conceptRelationships;

// 复杂 Selectors
export const selectConceptTrends = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.statistics?.statistics.trends;

export const selectAverageCognitiveValues = (state: { cognitive: CognitiveSliceState }) => 
  state.cognitive.statistics?.statistics.averageValues;

export const selectRecentCognitiveStates = (minutes: number = 30) => 
  (state: { cognitive: CognitiveSliceState }) => {
    const cutoffTime = Date.now() - (minutes * 60 * 1000);
    return state.cognitive.history.filter(s => s.timestamp >= cutoffTime);
  };

export default cognitiveSlice.reducer;