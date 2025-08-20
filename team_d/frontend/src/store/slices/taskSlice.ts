/**
 * 任务状态管理 Slice
 * 处理实验任务、会话管理、提交记录等
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { ExperimentTask, TaskSession, TaskSubmission, PerformanceMetrics } from '@/types';
import { STORAGE_KEYS } from '@/constants';
import * as taskService from '@/services/taskService';

// 状态类型定义
interface TaskState {
  // 可用任务列表
  availableTasks: ExperimentTask[];
  
  // 当前任务
  currentTask: ExperimentTask | null;
  
  // 当前会话
  currentSession: TaskSession | null;
  
  // 会话历史
  sessionHistory: TaskSession[];
  
  // 任务提交记录
  submissions: TaskSubmission[];
  
  // 加载状态
  loading: {
    tasks: boolean;
    session: boolean;
    submission: boolean;
    analysis: boolean;
  };
  
  // 错误状态
  error: string | null;
  
  // 任务统计
  statistics: {
    totalTasksCompleted: number;
    averageScore: number;
    totalWritingTime: number;
    improvementRate: number;
    streakDays: number;
  };
  
  // 性能分析
  performanceAnalysis: {
    strengths: string[];
    weaknesses: string[];
    recommendations: string[];
    trends: {
      concept: string;
      direction: 'improving' | 'declining' | 'stable';
      change: number;
    }[];
  } | null;
}

// 初始状态
const initialState: TaskState = {
  availableTasks: [],
  currentTask: null,
  currentSession: null,
  sessionHistory: [],
  submissions: [],
  loading: {
    tasks: false,
    session: false,
    submission: false,
    analysis: false,
  },
  error: null,
  statistics: {
    totalTasksCompleted: 0,
    averageScore: 0,
    totalWritingTime: 0,
    improvementRate: 0,
    streakDays: 0,
  },
  performanceAnalysis: null,
};

// ============ 异步 Actions ============

/** 获取可用任务列表 */
export const fetchAvailableTasksAsync = createAsyncThunk(
  'task/fetchAvailableTasks',
  async (params: {
    userId: string;
    difficulty?: 'easy' | 'medium' | 'hard';
    type?: 'writing' | 'comprehension' | 'creativity';
    limit?: number;
  }) => {
    const response = await taskService.getAvailableTasks(params);
    return response;
  }
);

/** 获取任务详情 */
export const fetchTaskDetailsAsync = createAsyncThunk(
  'task/fetchTaskDetails',
  async (taskId: string) => {
    const response = await taskService.getTaskDetails(taskId);
    return response;
  }
);

/** 开始新任务会话 */
export const startTaskSessionAsync = createAsyncThunk(
  'task/startSession',
  async (params: {
    userId: string;
    taskId: string;
    contextData?: any;
  }) => {
    const response = await taskService.startTaskSession(params);
    
    // 保存到本地存储
    localStorage.setItem(STORAGE_KEYS.SESSION_DATA, JSON.stringify(response));
    
    return response;
  }
);

/** 更新任务会话 */
export const updateTaskSessionAsync = createAsyncThunk(
  'task/updateSession',
  async (params: {
    sessionId: string;
    updates: Partial<TaskSession>;
  }) => {
    const response = await taskService.updateTaskSession(params);
    
    // 更新本地存储
    localStorage.setItem(STORAGE_KEYS.SESSION_DATA, JSON.stringify(response));
    
    return response;
  }
);

/** 提交任务答案 */
export const submitTaskAsync = createAsyncThunk(
  'task/submitTask',
  async (params: {
    sessionId: string;
    content: string;
    finalCognitiveState: any;
    metadata?: any;
  }) => {
    const response = await taskService.submitTask(params);
    
    // 清除本地存储的会话数据
    localStorage.removeItem(STORAGE_KEYS.SESSION_DATA);
    
    return response;
  }
);

/** 结束任务会话 */
export const endTaskSessionAsync = createAsyncThunk(
  'task/endSession',
  async (sessionId: string) => {
    const response = await taskService.endTaskSession(sessionId);
    
    // 清除本地存储
    localStorage.removeItem(STORAGE_KEYS.SESSION_DATA);
    
    return response;
  }
);

/** 获取会话历史 */
export const fetchSessionHistoryAsync = createAsyncThunk(
  'task/fetchSessionHistory',
  async (params: {
    userId: string;
    limit?: number;
    offset?: number;
    status?: 'active' | 'completed' | 'aborted';
  }) => {
    const response = await taskService.getSessionHistory(params);
    return response;
  }
);

/** 获取性能分析 */
export const fetchPerformanceAnalysisAsync = createAsyncThunk(
  'task/fetchPerformanceAnalysis',
  async (params: {
    userId: string;
    timeRange?: {
      start: number;
      end: number;
    };
    analysisType?: 'overview' | 'detailed' | 'trends';
  }) => {
    const response = await taskService.getPerformanceAnalysis(params);
    return response;
  }
);

// ============ Slice 定义 ============

const taskSlice = createSlice({
  name: 'task',
  initialState,
  reducers: {
    // 清除错误状态
    clearError: (state) => {
      state.error = null;
    },
    
    // 设置当前任务
    setCurrentTask: (state, action: PayloadAction<ExperimentTask | null>) => {
      state.currentTask = action.payload;
    },
    
    // 更新当前会话（本地状态）
    updateCurrentSessionLocal: (state, action: PayloadAction<Partial<TaskSession>>) => {
      if (state.currentSession) {
        state.currentSession = { ...state.currentSession, ...action.payload };
        
        // 更新本地存储
        localStorage.setItem(STORAGE_KEYS.SESSION_DATA, JSON.stringify(state.currentSession));
      }
    },
    
    // 添加行为数据到当前会话
    addBehaviorDataToSession: (state, action: PayloadAction<any>) => {
      if (state.currentSession) {
        state.currentSession.behaviorData.push(action.payload);
        
        // 更新本地存储
        localStorage.setItem(STORAGE_KEYS.SESSION_DATA, JSON.stringify(state.currentSession));
      }
    },
    
    // 添加认知状态到当前会话
    addCognitiveStateToSession: (state, action: PayloadAction<any>) => {
      if (state.currentSession) {
        state.currentSession.cognitiveStates.push(action.payload);
        
        // 更新本地存储
        localStorage.setItem(STORAGE_KEYS.SESSION_DATA, JSON.stringify(state.currentSession));
      }
    },
    
    // 设置加载状态
    setLoading: (state, action: PayloadAction<{ type: keyof TaskState['loading']; loading: boolean }>) => {
      const { type, loading } = action.payload;
      state.loading[type] = loading;
    },
    
    // 添加提交记录
    addSubmission: (state, action: PayloadAction<TaskSubmission>) => {
      const newSubmission = action.payload;
      
      // 检查是否已存在
      const existingIndex = state.submissions.findIndex(s => s.id === newSubmission.id);
      
      if (existingIndex !== -1) {
        state.submissions[existingIndex] = newSubmission;
      } else {
        state.submissions.unshift(newSubmission);
      }
      
      // 限制提交记录数量
      if (state.submissions.length > 50) {
        state.submissions = state.submissions.slice(0, 50);
      }
      
      // 更新统计信息
      updateTaskStatistics(state);
    },
    
    // 从本地存储恢复会话
    restoreSessionFromStorage: (state) => {
      const sessionData = localStorage.getItem(STORAGE_KEYS.SESSION_DATA);
      if (sessionData) {
        try {
          const session = JSON.parse(sessionData);
          state.currentSession = session;
          
          // 查找对应的任务
          const task = state.availableTasks.find(t => t.id === session.taskId);
          if (task) {
            state.currentTask = task;
          }
        } catch (error) {
          console.warn('Failed to restore session from storage:', error);
          localStorage.removeItem(STORAGE_KEYS.SESSION_DATA);
        }
      }
    },
    
    // 清除当前会话
    clearCurrentSession: (state) => {
      state.currentSession = null;
      state.currentTask = null;
      localStorage.removeItem(STORAGE_KEYS.SESSION_DATA);
    },
    
    // 重置任务状态
    resetTaskState: (state) => {
      Object.assign(state, initialState);
      localStorage.removeItem(STORAGE_KEYS.SESSION_DATA);
    },
  },
  extraReducers: (builder) => {
    // ====== 获取可用任务 ======
    builder
      .addCase(fetchAvailableTasksAsync.pending, (state) => {
        state.loading.tasks = true;
        state.error = null;
      })
      .addCase(fetchAvailableTasksAsync.fulfilled, (state, action) => {
        state.loading.tasks = false;
        state.availableTasks = action.payload;
      })
      .addCase(fetchAvailableTasksAsync.rejected, (state, action) => {
        state.loading.tasks = false;
        state.error = action.error.message || '获取任务列表失败';
      });
    
    // ====== 获取任务详情 ======
    builder
      .addCase(fetchTaskDetailsAsync.pending, (state) => {
        state.loading.tasks = true;
        state.error = null;
      })
      .addCase(fetchTaskDetailsAsync.fulfilled, (state, action) => {
        state.loading.tasks = false;
        state.currentTask = action.payload;
      })
      .addCase(fetchTaskDetailsAsync.rejected, (state, action) => {
        state.loading.tasks = false;
        state.error = action.error.message || '获取任务详情失败';
      });
    
    // ====== 开始任务会话 ======
    builder
      .addCase(startTaskSessionAsync.pending, (state) => {
        state.loading.session = true;
        state.error = null;
      })
      .addCase(startTaskSessionAsync.fulfilled, (state, action) => {
        state.loading.session = false;
        state.currentSession = action.payload;
        
        // 将会话添加到历史记录
        const existingIndex = state.sessionHistory.findIndex(s => s.id === action.payload.id);
        if (existingIndex === -1) {
          state.sessionHistory.unshift(action.payload);
        }
      })
      .addCase(startTaskSessionAsync.rejected, (state, action) => {
        state.loading.session = false;
        state.error = action.error.message || '开始任务会话失败';
      });
    
    // ====== 更新任务会话 ======
    builder
      .addCase(updateTaskSessionAsync.pending, (state) => {
        state.loading.session = true;
      })
      .addCase(updateTaskSessionAsync.fulfilled, (state, action) => {
        state.loading.session = false;
        state.currentSession = action.payload;
        
        // 更新历史记录中的会话
        const historyIndex = state.sessionHistory.findIndex(s => s.id === action.payload.id);
        if (historyIndex !== -1) {
          state.sessionHistory[historyIndex] = action.payload;
        }
      })
      .addCase(updateTaskSessionAsync.rejected, (state, action) => {
        state.loading.session = false;
        state.error = action.error.message || '更新任务会话失败';
      });
    
    // ====== 提交任务 ======
    builder
      .addCase(submitTaskAsync.pending, (state) => {
        state.loading.submission = true;
        state.error = null;
      })
      .addCase(submitTaskAsync.fulfilled, (state, action) => {
        state.loading.submission = false;
        
        // 添加提交记录
        state.submissions.unshift(action.payload.submission);
        
        // 更新会话状态
        if (state.currentSession) {
          state.currentSession.status = 'completed';
          state.currentSession.endTime = Date.now();
          state.currentSession.submissions.push(action.payload.submission);
          state.currentSession.performance = action.payload.performance;
        }
        
        // 更新统计信息
        updateTaskStatistics(state);
      })
      .addCase(submitTaskAsync.rejected, (state, action) => {
        state.loading.submission = false;
        state.error = action.error.message || '提交任务失败';
      });
    
    // ====== 结束任务会话 ======
    builder
      .addCase(endTaskSessionAsync.pending, (state) => {
        state.loading.session = true;
      })
      .addCase(endTaskSessionAsync.fulfilled, (state, action) => {
        state.loading.session = false;
        
        // 更新会话状态
        if (state.currentSession) {
          state.currentSession.status = action.payload.status;
          state.currentSession.endTime = action.payload.endTime;
        }
        
        // 清除当前会话和任务
        state.currentSession = null;
        state.currentTask = null;
      })
      .addCase(endTaskSessionAsync.rejected, (state, action) => {
        state.loading.session = false;
        state.error = action.error.message || '结束任务会话失败';
      });
    
    // ====== 获取会话历史 ======
    builder
      .addCase(fetchSessionHistoryAsync.pending, (state) => {
        state.loading.session = true;
        state.error = null;
      })
      .addCase(fetchSessionHistoryAsync.fulfilled, (state, action) => {
        state.loading.session = false;
        state.sessionHistory = action.payload.sessions;
        updateTaskStatistics(state);
      })
      .addCase(fetchSessionHistoryAsync.rejected, (state, action) => {
        state.loading.session = false;
        state.error = action.error.message || '获取会话历史失败';
      });
    
    // ====== 获取性能分析 ======
    builder
      .addCase(fetchPerformanceAnalysisAsync.pending, (state) => {
        state.loading.analysis = true;
        state.error = null;
      })
      .addCase(fetchPerformanceAnalysisAsync.fulfilled, (state, action) => {
        state.loading.analysis = false;
        state.performanceAnalysis = action.payload.analysis;
        
        if (action.payload.statistics) {
          state.statistics = { ...state.statistics, ...action.payload.statistics };
        }
      })
      .addCase(fetchPerformanceAnalysisAsync.rejected, (state, action) => {
        state.loading.analysis = false;
        state.error = action.error.message || '获取性能分析失败';
      });
  },
});

// ============ 辅助函数 ============

/** 更新任务统计信息 */
function updateTaskStatistics(state: TaskState) {
  const completedSessions = state.sessionHistory.filter(s => s.status === 'completed');
  
  // 完成的任务数量
  state.statistics.totalTasksCompleted = completedSessions.length;
  
  // 平均分数
  const scoredSessions = completedSessions.filter(s => s.performance);
  if (scoredSessions.length > 0) {
    const totalScore = scoredSessions.reduce((sum, session) => {
      return sum + (session.performance?.writingQuality || 0);
    }, 0);
    state.statistics.averageScore = totalScore / scoredSessions.length;
  }
  
  // 总写作时间
  state.statistics.totalWritingTime = completedSessions.reduce((total, session) => {
    if (session.endTime && session.startTime) {
      return total + (session.endTime - session.startTime) / (1000 * 60); // 转换为分钟
    }
    return total;
  }, 0);
  
  // 改进率计算
  if (scoredSessions.length >= 2) {
    const recentSessions = scoredSessions.slice(0, 5); // 最近5次
    const earlierSessions = scoredSessions.slice(-5); // 较早的5次
    
    if (recentSessions.length > 0 && earlierSessions.length > 0) {
      const recentAvg = recentSessions.reduce((sum, s) => sum + (s.performance?.writingQuality || 0), 0) / recentSessions.length;
      const earlierAvg = earlierSessions.reduce((sum, s) => sum + (s.performance?.writingQuality || 0), 0) / earlierSessions.length;
      
      state.statistics.improvementRate = ((recentAvg - earlierAvg) / earlierAvg) * 100;
    }
  }
  
  // 连续天数计算
  const today = new Date();
  let streakDays = 0;
  let currentDate = new Date(today);
  
  while (streakDays < 365) { // 最多计算365天
    const dayStart = new Date(currentDate);
    dayStart.setHours(0, 0, 0, 0);
    const dayEnd = new Date(currentDate);
    dayEnd.setHours(23, 59, 59, 999);
    
    const hasActivityOnDay = completedSessions.some(session => {
      const sessionDate = new Date(session.startTime);
      return sessionDate >= dayStart && sessionDate <= dayEnd;
    });
    
    if (hasActivityOnDay) {
      streakDays++;
      currentDate.setDate(currentDate.getDate() - 1);
    } else {
      break;
    }
  }
  
  state.statistics.streakDays = streakDays;
}

// ============ 导出 Actions 和 Selectors ============

export const {
  clearError,
  setCurrentTask,
  updateCurrentSessionLocal,
  addBehaviorDataToSession,
  addCognitiveStateToSession,
  setLoading,
  addSubmission,
  restoreSessionFromStorage,
  clearCurrentSession,
  resetTaskState,
} = taskSlice.actions;

// Selectors
export const selectTask = (state: { task: TaskState }) => state.task;
export const selectAvailableTasks = (state: { task: TaskState }) => state.task.availableTasks;
export const selectCurrentTask = (state: { task: TaskState }) => state.task.currentTask;
export const selectCurrentTaskSession = (state: { task: TaskState }) => state.task.currentSession;
export const selectSessionHistory = (state: { task: TaskState }) => state.task.sessionHistory;
export const selectTaskSubmissions = (state: { task: TaskState }) => state.task.submissions;
export const selectTaskLoading = (state: { task: TaskState }) => state.task.loading;
export const selectTaskError = (state: { task: TaskState }) => state.task.error;
export const selectTaskStatistics = (state: { task: TaskState }) => state.task.statistics;
export const selectPerformanceAnalysis = (state: { task: TaskState }) => state.task.performanceAnalysis;

// 复杂 Selectors
export const selectActiveSession = (state: { task: TaskState }) => 
  state.task.sessionHistory.find(s => s.status === 'active') || state.task.currentSession;

export const selectCompletedSessions = (state: { task: TaskState }) => 
  state.task.sessionHistory.filter(s => s.status === 'completed');

export const selectRecentSubmissions = (days: number = 7) => 
  (state: { task: TaskState }) => {
    const cutoffTime = Date.now() - (days * 24 * 60 * 60 * 1000);
    return state.task.submissions.filter(s => s.timestamp >= cutoffTime);
  };

export const selectTasksByType = (type: 'writing' | 'comprehension' | 'creativity') => 
  (state: { task: TaskState }) => state.task.availableTasks.filter(t => t.type === type);

export const selectTasksByDifficulty = (difficulty: 'easy' | 'medium' | 'hard') => 
  (state: { task: TaskState }) => state.task.availableTasks.filter(t => t.difficulty === difficulty);

export const selectIsSessionActive = (state: { task: TaskState }) => 
  state.task.currentSession?.status === 'active';

export default taskSlice.reducer;