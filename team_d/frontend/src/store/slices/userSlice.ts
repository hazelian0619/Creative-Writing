/**
 * 用户状态管理 Slice
 * 处理用户个人信息、偏好设置、会话记录等
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { UserProfile, UserPreferences, TaskSession } from '@/types';
import { STORAGE_KEYS, DEFAULT_PREFERENCES } from '@/constants';
import * as userService from '@/services/userService';

// 状态类型定义
interface UserState {
  // 用户个人资料
  profile: UserProfile | null;
  
  // 用户偏好设置
  preferences: UserPreferences;
  
  // 用户会话记录
  sessions: TaskSession[];
  
  // 当前活跃会话
  currentSession: TaskSession | null;
  
  // 加载状态
  loading: {
    profile: boolean;
    preferences: boolean;
    sessions: boolean;
  };
  
  // 错误状态
  error: string | null;
  
  // 统计信息
  statistics: {
    totalSessions: number;
    totalWritingTime: number; // 总写作时间（分钟）
    averagePerformance: number; // 平均性能评分
    lastActiveDate: string | null;
    weeklyProgress: number[]; // 每周进度（7天）
  };
  
  // 成就和徽章
  achievements: {
    id: string;
    name: string;
    description: string;
    icon: string;
    unlockedAt: string;
    progress?: number; // 进度百分比
  }[];
}

// 初始状态
const initialState: UserState = {
  profile: null,
  preferences: {
    ...DEFAULT_PREFERENCES,
    // 从本地存储加载偏好设置
    ...JSON.parse(localStorage.getItem(STORAGE_KEYS.USER_PREFERENCES) || '{}'),
  },
  sessions: [],
  currentSession: null,
  loading: {
    profile: false,
    preferences: false,
    sessions: false,
  },
  error: null,
  statistics: {
    totalSessions: 0,
    totalWritingTime: 0,
    averagePerformance: 0,
    lastActiveDate: null,
    weeklyProgress: [0, 0, 0, 0, 0, 0, 0],
  },
  achievements: [],
};

// ============ 异步 Actions ============

/** 获取用户个人资料 */
export const fetchUserProfileAsync = createAsyncThunk(
  'user/fetchProfile',
  async (userId: string) => {
    const response = await userService.getUserProfile(userId);
    return response;
  }
);

/** 更新用户个人资料 */
export const updateUserProfileAsync = createAsyncThunk(
  'user/updateProfile',
  async (profileData: Partial<UserProfile>) => {
    const response = await userService.updateUserProfile(profileData);
    return response;
  }
);

/** 获取用户偏好设置 */
export const fetchUserPreferencesAsync = createAsyncThunk(
  'user/fetchPreferences',
  async (userId: string) => {
    const response = await userService.getUserPreferences(userId);
    return response;
  }
);

/** 更新用户偏好设置 */
export const updateUserPreferencesAsync = createAsyncThunk(
  'user/updatePreferences',
  async (preferences: Partial<UserPreferences>) => {
    const response = await userService.updateUserPreferences(preferences);
    
    // 同时更新本地存储
    const currentPrefs = JSON.parse(localStorage.getItem(STORAGE_KEYS.USER_PREFERENCES) || '{}');
    const updatedPrefs = { ...currentPrefs, ...preferences };
    localStorage.setItem(STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(updatedPrefs));
    
    return response;
  }
);

/** 获取用户会话历史 */
export const fetchUserSessionsAsync = createAsyncThunk(
  'user/fetchSessions',
  async (params: {
    userId: string;
    limit?: number;
    offset?: number;
    timeRange?: {
      start: number;
      end: number;
    };
  }) => {
    const response = await userService.getUserSessions(params);
    return response;
  }
);

/** 获取用户统计信息 */
export const fetchUserStatisticsAsync = createAsyncThunk(
  'user/fetchStatistics',
  async (userId: string) => {
    const response = await userService.getUserStatistics(userId);
    return response;
  }
);

/** 获取用户成就 */
export const fetchUserAchievementsAsync = createAsyncThunk(
  'user/fetchAchievements',
  async (userId: string) => {
    const response = await userService.getUserAchievements(userId);
    return response;
  }
);

// ============ Slice 定义 ============

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    // 清除错误状态
    clearError: (state) => {
      state.error = null;
    },
    
    // 更新偏好设置（本地）
    updatePreferences: (state, action: PayloadAction<Partial<UserPreferences>>) => {
      state.preferences = { ...state.preferences, ...action.payload };
      localStorage.setItem(STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(state.preferences));
    },
    
    // 设置当前会话
    setCurrentSession: (state, action: PayloadAction<TaskSession | null>) => {
      state.currentSession = action.payload;
      
      if (action.payload) {
        // 将当前会话保存到本地存储
        localStorage.setItem(STORAGE_KEYS.SESSION_DATA, JSON.stringify(action.payload));
      } else {
        localStorage.removeItem(STORAGE_KEYS.SESSION_DATA);
      }
    },
    
    // 更新当前会话
    updateCurrentSession: (state, action: PayloadAction<Partial<TaskSession>>) => {
      if (state.currentSession) {
        state.currentSession = { ...state.currentSession, ...action.payload };
        localStorage.setItem(STORAGE_KEYS.SESSION_DATA, JSON.stringify(state.currentSession));
      }
    },
    
    // 添加新会话到历史记录
    addSession: (state, action: PayloadAction<TaskSession>) => {
      const newSession = action.payload;
      
      // 检查是否已存在
      const existingIndex = state.sessions.findIndex(s => s.id === newSession.id);
      
      if (existingIndex !== -1) {
        // 更新现有会话
        state.sessions[existingIndex] = newSession;
      } else {
        // 添加新会话
        state.sessions.unshift(newSession);
      }
      
      // 限制会话记录数量
      if (state.sessions.length > 100) {
        state.sessions = state.sessions.slice(0, 100);
      }
      
      // 更新统计信息
      updateUserStatistics(state);
    },
    
    // 更新会话状态
    updateSession: (state, action: PayloadAction<{ id: string; updates: Partial<TaskSession> }>) => {
      const { id, updates } = action.payload;
      const sessionIndex = state.sessions.findIndex(s => s.id === id);
      
      if (sessionIndex !== -1) {
        state.sessions[sessionIndex] = { ...state.sessions[sessionIndex], ...updates };
        updateUserStatistics(state);
      }
      
      // 如果是当前会话，也要更新
      if (state.currentSession && state.currentSession.id === id) {
        state.currentSession = { ...state.currentSession, ...updates };
      }
    },
    
    // 设置加载状态
    setLoading: (state, action: PayloadAction<{ type: keyof UserState['loading']; loading: boolean }>) => {
      const { type, loading } = action.payload;
      state.loading[type] = loading;
    },
    
    // 解锁新成就
    unlockAchievement: (state, action: PayloadAction<{
      id: string;
      name: string;
      description: string;
      icon: string;
    }>) => {
      const achievement = {
        ...action.payload,
        unlockedAt: new Date().toISOString(),
      };
      
      // 检查是否已解锁
      const existingIndex = state.achievements.findIndex(a => a.id === achievement.id);
      
      if (existingIndex === -1) {
        state.achievements.unshift(achievement);
      }
    },
    
    // 更新成就进度
    updateAchievementProgress: (state, action: PayloadAction<{ id: string; progress: number }>) => {
      const { id, progress } = action.payload;
      const achievementIndex = state.achievements.findIndex(a => a.id === id);
      
      if (achievementIndex !== -1) {
        state.achievements[achievementIndex].progress = progress;
      }
    },
    
    // 重置用户状态
    resetUserState: (state) => {
      Object.assign(state, {
        ...initialState,
        preferences: state.preferences, // 保持偏好设置
      });
    },
  },
  extraReducers: (builder) => {
    // ====== 获取用户资料 ======
    builder
      .addCase(fetchUserProfileAsync.pending, (state) => {
        state.loading.profile = true;
        state.error = null;
      })
      .addCase(fetchUserProfileAsync.fulfilled, (state, action) => {
        state.loading.profile = false;
        state.profile = action.payload;
      })
      .addCase(fetchUserProfileAsync.rejected, (state, action) => {
        state.loading.profile = false;
        state.error = action.error.message || '获取用户资料失败';
      });
    
    // ====== 更新用户资料 ======
    builder
      .addCase(updateUserProfileAsync.pending, (state) => {
        state.loading.profile = true;
        state.error = null;
      })
      .addCase(updateUserProfileAsync.fulfilled, (state, action) => {
        state.loading.profile = false;
        state.profile = action.payload;
      })
      .addCase(updateUserProfileAsync.rejected, (state, action) => {
        state.loading.profile = false;
        state.error = action.error.message || '更新用户资料失败';
      });
    
    // ====== 获取偏好设置 ======
    builder
      .addCase(fetchUserPreferencesAsync.pending, (state) => {
        state.loading.preferences = true;
        state.error = null;
      })
      .addCase(fetchUserPreferencesAsync.fulfilled, (state, action) => {
        state.loading.preferences = false;
        state.preferences = action.payload;
        localStorage.setItem(STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(action.payload));
      })
      .addCase(fetchUserPreferencesAsync.rejected, (state, action) => {
        state.loading.preferences = false;
        state.error = action.error.message || '获取偏好设置失败';
      });
    
    // ====== 更新偏好设置 ======
    builder
      .addCase(updateUserPreferencesAsync.pending, (state) => {
        state.loading.preferences = true;
        state.error = null;
      })
      .addCase(updateUserPreferencesAsync.fulfilled, (state, action) => {
        state.loading.preferences = false;
        state.preferences = action.payload;
      })
      .addCase(updateUserPreferencesAsync.rejected, (state, action) => {
        state.loading.preferences = false;
        state.error = action.error.message || '更新偏好设置失败';
      });
    
    // ====== 获取会话历史 ======
    builder
      .addCase(fetchUserSessionsAsync.pending, (state) => {
        state.loading.sessions = true;
        state.error = null;
      })
      .addCase(fetchUserSessionsAsync.fulfilled, (state, action) => {
        state.loading.sessions = false;
        state.sessions = action.payload.sessions;
        updateUserStatistics(state);
      })
      .addCase(fetchUserSessionsAsync.rejected, (state, action) => {
        state.loading.sessions = false;
        state.error = action.error.message || '获取会话历史失败';
      });
    
    // ====== 获取统计信息 ======
    builder
      .addCase(fetchUserStatisticsAsync.fulfilled, (state, action) => {
        state.statistics = action.payload;
      });
    
    // ====== 获取成就 ======
    builder
      .addCase(fetchUserAchievementsAsync.fulfilled, (state, action) => {
        state.achievements = action.payload;
      });
  },
});

// ============ 辅助函数 ============

/** 更新用户统计信息 */
function updateUserStatistics(state: UserState) {
  const sessions = state.sessions;
  
  // 总会话数
  state.statistics.totalSessions = sessions.length;
  
  // 总写作时间
  state.statistics.totalWritingTime = sessions.reduce((total, session) => {
    if (session.endTime && session.startTime) {
      return total + (session.endTime - session.startTime) / (1000 * 60); // 转换为分钟
    }
    return total;
  }, 0);
  
  // 平均性能评分
  const completedSessions = sessions.filter(s => s.performance);
  if (completedSessions.length > 0) {
    const totalScore = completedSessions.reduce((sum, session) => {
      return sum + (session.performance?.writingQuality || 0);
    }, 0);
    state.statistics.averagePerformance = totalScore / completedSessions.length;
  }
  
  // 最后活跃日期
  if (sessions.length > 0) {
    const latestSession = sessions.reduce((latest, session) => {
      return session.startTime > latest.startTime ? session : latest;
    });
    state.statistics.lastActiveDate = new Date(latestSession.startTime).toISOString();
  }
  
  // 计算每周进度
  const oneWeekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
  const weekSessions = sessions.filter(s => s.startTime >= oneWeekAgo);
  
  // 按天分组
  const dailyProgress = Array(7).fill(0);
  weekSessions.forEach(session => {
    const dayIndex = Math.floor((Date.now() - session.startTime) / (24 * 60 * 60 * 1000));
    if (dayIndex >= 0 && dayIndex < 7) {
      dailyProgress[6 - dayIndex] += 1; // 倒序，最近的一天在右侧
    }
  });
  
  state.statistics.weeklyProgress = dailyProgress;
}

// ============ 导出 Actions 和 Selectors ============

export const {
  clearError,
  updatePreferences,
  setCurrentSession,
  updateCurrentSession,
  addSession,
  updateSession,
  setLoading,
  unlockAchievement,
  updateAchievementProgress,
  resetUserState,
} = userSlice.actions;

// Selectors
export const selectUser = (state: { user: UserState }) => state.user;
export const selectUserProfile = (state: { user: UserState }) => state.user.profile;
export const selectUserPreferences = (state: { user: UserState }) => state.user.preferences;
export const selectUserSessions = (state: { user: UserState }) => state.user.sessions;
export const selectCurrentSession = (state: { user: UserState }) => state.user.currentSession;
export const selectUserLoading = (state: { user: UserState }) => state.user.loading;
export const selectUserError = (state: { user: UserState }) => state.user.error;
export const selectUserStatistics = (state: { user: UserState }) => state.user.statistics;
export const selectUserAchievements = (state: { user: UserState }) => state.user.achievements;

// 复杂 Selectors
export const selectRecentSessions = (days: number = 7) => 
  (state: { user: UserState }) => {
    const cutoffTime = Date.now() - (days * 24 * 60 * 60 * 1000);
    return state.user.sessions.filter(s => s.startTime >= cutoffTime);
  };

export const selectCompletedSessions = (state: { user: UserState }) => 
  state.user.sessions.filter(s => s.status === 'completed');

export const selectActiveSession = (state: { user: UserState }) => 
  state.user.sessions.find(s => s.status === 'active') || state.user.currentSession;

export const selectUnlockedAchievements = (state: { user: UserState }) => 
  state.user.achievements.filter(a => !a.progress || a.progress >= 100);

export const selectProgressingAchievements = (state: { user: UserState }) => 
  state.user.achievements.filter(a => a.progress && a.progress < 100);

export default userSlice.reducer;