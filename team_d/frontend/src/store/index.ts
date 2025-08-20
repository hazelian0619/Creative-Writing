/**
 * Redux Store 配置
 * 使用 Redux Toolkit 进行状态管理
 */

import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';

// 导入各个 slice
import authSlice from './slices/authSlice';
import userSlice from './slices/userSlice';
import cognitiveSlice from './slices/cognitiveSlice';
import taskSlice from './slices/taskSlice';
import uiSlice from './slices/uiSlice';

// 配置 store
export const store = configureStore({
  reducer: {
    auth: authSlice,
    user: userSlice,
    cognitive: cognitiveSlice,
    task: taskSlice,
    ui: uiSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // 忽略这些 action types 的序列化检查
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
        // 忽略这些 field paths 的序列化检查
        ignoredPaths: ['register'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
});

// 类型定义
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// 类型化的 hooks
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

// 导出 store 实例
export default store;