/**
 * UI状态管理 Slice
 * 处理界面相关的状态，如加载状态、通知、主题等
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { SystemNotificationMessage } from '@/types';
import { STORAGE_KEYS, DEFAULT_PREFERENCES } from '@/constants';

// 状态类型定义
interface UIState {
  // 侧边栏状态
  sidebarCollapsed: boolean;
  
  // 主题设置
  theme: 'light' | 'dark';
  
  // 全局加载状态
  loading: {
    global: boolean;
    cognitive: boolean;
    inference: boolean;
    export: boolean;
  };
  
  // 系统通知
  notifications: SystemNotificationMessage[];
  
  // 模态框状态
  modals: {
    settingsVisible: boolean;
    exportVisible: boolean;
    helpVisible: boolean;
  };
  
  // 页面状态
  currentPage: string;
  pageHistory: string[];
  
  // 错误边界
  error: {
    hasError: boolean;
    errorMessage: string | null;
    errorStack: string | null;
  };
  
  // 网络状态
  networkStatus: 'online' | 'offline' | 'slow';
  
  // 屏幕尺寸
  screenSize: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'xxl';
}

// 初始状态
const initialState: UIState = {
  sidebarCollapsed: localStorage.getItem(STORAGE_KEYS.SIDEBAR_COLLAPSED) === 'true',
  theme: (localStorage.getItem(STORAGE_KEYS.THEME) as 'light' | 'dark') || DEFAULT_PREFERENCES.theme,
  loading: {
    global: false,
    cognitive: false,
    inference: false,
    export: false,
  },
  notifications: [],
  modals: {
    settingsVisible: false,
    exportVisible: false,
    helpVisible: false,
  },
  currentPage: localStorage.getItem(STORAGE_KEYS.LAST_VISITED_PAGE) || '/',
  pageHistory: [],
  error: {
    hasError: false,
    errorMessage: null,
    errorStack: null,
  },
  networkStatus: 'online',
  screenSize: 'lg',
};

// ============ Slice 定义 ============

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    // 切换侧边栏
    toggleSidebar: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
      localStorage.setItem(STORAGE_KEYS.SIDEBAR_COLLAPSED, String(state.sidebarCollapsed));
    },
    
    // 设置侧边栏状态
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload;
      localStorage.setItem(STORAGE_KEYS.SIDEBAR_COLLAPSED, String(action.payload));
    },
    
    // 切换主题
    toggleTheme: (state) => {
      state.theme = state.theme === 'light' ? 'dark' : 'light';
      localStorage.setItem(STORAGE_KEYS.THEME, state.theme);
    },
    
    // 设置主题
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
      localStorage.setItem(STORAGE_KEYS.THEME, action.payload);
    },
    
    // 设置全局加载状态
    setGlobalLoading: (state, action: PayloadAction<boolean>) => {
      state.loading.global = action.payload;
    },
    
    // 设置特定类型的加载状态
    setLoading: (state, action: PayloadAction<{ type: keyof UIState['loading']; loading: boolean }>) => {
      const { type, loading } = action.payload;
      state.loading[type] = loading;
    },
    
    // 添加通知
    addNotification: (state, action: PayloadAction<Omit<SystemNotificationMessage, 'timestamp'>>) => {
      const notification: SystemNotificationMessage = {
        ...action.payload,
        timestamp: Date.now(),
      };
      
      state.notifications.unshift(notification);
      
      // 限制通知数量
      if (state.notifications.length > 10) {
        state.notifications = state.notifications.slice(0, 10);
      }
    },
    
    // 移除通知
    removeNotification: (state, action: PayloadAction<number>) => {
      const timestamp = action.payload;
      state.notifications = state.notifications.filter(n => n.timestamp !== timestamp);
    },
    
    // 清空所有通知
    clearNotifications: (state) => {
      state.notifications = [];
    },
    
    // 显示模态框
    showModal: (state, action: PayloadAction<keyof UIState['modals']>) => {
      const modalType = action.payload;
      state.modals[modalType] = true;
    },
    
    // 隐藏模态框
    hideModal: (state, action: PayloadAction<keyof UIState['modals']>) => {
      const modalType = action.payload;
      state.modals[modalType] = false;
    },
    
    // 隐藏所有模态框
    hideAllModals: (state) => {
      Object.keys(state.modals).forEach(key => {
        state.modals[key as keyof UIState['modals']] = false;
      });
    },
    
    // 设置当前页面
    setCurrentPage: (state, action: PayloadAction<string>) => {
      const newPage = action.payload;
      
      // 更新页面历史
      if (state.currentPage !== newPage) {
        state.pageHistory.unshift(state.currentPage);
        
        // 限制历史记录数量
        if (state.pageHistory.length > 10) {
          state.pageHistory = state.pageHistory.slice(0, 10);
        }
      }
      
      state.currentPage = newPage;
      localStorage.setItem(STORAGE_KEYS.LAST_VISITED_PAGE, newPage);
    },
    
    // 设置错误状态
    setError: (state, action: PayloadAction<{ message: string; stack?: string }>) => {
      state.error = {
        hasError: true,
        errorMessage: action.payload.message,
        errorStack: action.payload.stack || null,
      };
    },
    
    // 清除错误状态
    clearError: (state) => {
      state.error = {
        hasError: false,
        errorMessage: null,
        errorStack: null,
      };
    },
    
    // 设置网络状态
    setNetworkStatus: (state, action: PayloadAction<'online' | 'offline' | 'slow'>) => {
      state.networkStatus = action.payload;
      
      // 网络状态改变时添加通知
      if (action.payload === 'offline') {
        state.notifications.unshift({
          type: 'system_notification',
          timestamp: Date.now(),
          data: {
            level: 'warning',
            title: '网络连接中断',
            content: '当前网络连接不可用，部分功能可能受限',
            duration: 0, // 持续显示直到网络恢复
          },
        });
      } else if (action.payload === 'online' && state.networkStatus === 'offline') {
        // 移除离线通知
        state.notifications = state.notifications.filter(
          n => n.data.title !== '网络连接中断'
        );
        
        // 添加恢复通知
        state.notifications.unshift({
          type: 'system_notification',
          timestamp: Date.now(),
          data: {
            level: 'success',
            title: '网络连接已恢复',
            content: '网络连接已恢复正常',
            duration: 3000,
          },
        });
      }
    },
    
    // 设置屏幕尺寸
    setScreenSize: (state, action: PayloadAction<UIState['screenSize']>) => {
      state.screenSize = action.payload;
      
      // 在小屏幕下自动收起侧边栏
      if (['xs', 'sm'].includes(action.payload)) {
        state.sidebarCollapsed = true;
      }
    },
    
    // 重置UI状态
    resetUIState: (state) => {
      Object.assign(state, {
        ...initialState,
        sidebarCollapsed: state.sidebarCollapsed, // 保持侧边栏状态
        theme: state.theme, // 保持主题设置
      });
    },
  },
});

// ============ 导出 Actions 和 Selectors ============

export const {
  toggleSidebar,
  setSidebarCollapsed,
  toggleTheme,
  setTheme,
  setGlobalLoading,
  setLoading,
  addNotification,
  removeNotification,
  clearNotifications,
  showModal,
  hideModal,
  hideAllModals,
  setCurrentPage,
  setError,
  clearError,
  setNetworkStatus,
  setScreenSize,
  resetUIState,
} = uiSlice.actions;

// Selectors
export const selectUI = (state: { ui: UIState }) => state.ui;
export const selectSidebarCollapsed = (state: { ui: UIState }) => state.ui.sidebarCollapsed;
export const selectTheme = (state: { ui: UIState }) => state.ui.theme;
export const selectLoading = (state: { ui: UIState }) => state.ui.loading;
export const selectGlobalLoading = (state: { ui: UIState }) => state.ui.loading.global;
export const selectNotifications = (state: { ui: UIState }) => state.ui.notifications;
export const selectModals = (state: { ui: UIState }) => state.ui.modals;
export const selectCurrentPage = (state: { ui: UIState }) => state.ui.currentPage;
export const selectPageHistory = (state: { ui: UIState }) => state.ui.pageHistory;
export const selectError = (state: { ui: UIState }) => state.ui.error;
export const selectNetworkStatus = (state: { ui: UIState }) => state.ui.networkStatus;
export const selectScreenSize = (state: { ui: UIState }) => state.ui.screenSize;

// 复杂 Selectors
export const selectUnreadNotifications = (state: { ui: UIState }) => 
  state.ui.notifications.filter(n => !n.data.read);

export const selectIsAnyModalVisible = (state: { ui: UIState }) => 
  Object.values(state.ui.modals).some(visible => visible);

export const selectIsOffline = (state: { ui: UIState }) => 
  state.ui.networkStatus === 'offline';

export const selectIsMobile = (state: { ui: UIState }) => 
  ['xs', 'sm'].includes(state.ui.screenSize);

export default uiSlice.reducer;