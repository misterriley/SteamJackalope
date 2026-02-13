import React from 'react';
import AsyncSelect from 'react-select/async';
import { searchGames } from '../api';

interface SeedSelectorProps {
  selected: string[];
  onChange: (selected: string[]) => void;
  placeholder?: string;
}

const customStyles = {
  control: (base: any, state: any) => ({
    ...base,
    backgroundColor: '#1e293b',
    borderColor: state.isFocused ? '#eab308' : '#334155',
    boxShadow: state.isFocused ? '0 0 0 1px #eab308' : 'none',
    '&:hover': {
      borderColor: '#eab308',
    },
    borderRadius: '0.5rem',
    padding: '2px',
  }),
  menu: (base: any) => ({
    ...base,
    backgroundColor: '#1e293b',
    border: '1px solid #334155',
    zIndex: 9999,
  }),
  menuPortal: (base: any) => ({
    ...base,
    zIndex: 9999,
  }),
  option: (base: any, state: any) => ({
    ...base,
    backgroundColor: state.isFocused ? '#334155' : 'transparent',
    color: state.isSelected ? '#eab308' : '#f8fafc',
    '&:active': {
      backgroundColor: '#334155',
    },
  }),
  multiValue: (base: any) => ({
    ...base,
    backgroundColor: '#334155',
    borderRadius: '4px',
  }),
  multiValueLabel: (base: any) => ({
    ...base,
    color: '#f8fafc',
  }),
  multiValueRemove: (base: any) => ({
    ...base,
    color: '#94a3b8',
    '&:hover': {
      backgroundColor: '#ef4444',
      color: 'white',
    },
  }),
  input: (base: any) => ({
    ...base,
    color: '#f8fafc',
  }),
  placeholder: (base: any) => ({
    ...base,
    color: '#94a3b8',
  }),
  loadingIndicator: (base: any) => ({
    ...base,
    color: '#eab308',
  }),
};

const SeedSelector: React.FC<SeedSelectorProps> = ({ selected, onChange, placeholder }) => {
  const loadOptions = async (inputValue: string) => {
    if (inputValue.length < 2) return [];
    try {
      const results = await searchGames(inputValue);
      return results.map(name => ({ value: name, label: name }));
    } catch (error) {
      console.error("Search failed", error);
      return [];
    }
  };

  const value = selected.map(s => ({ value: s, label: s }));

  return (
    <AsyncSelect
      isMulti
      cacheOptions
      defaultOptions={false}
      loadOptions={loadOptions}
      value={value}
      onChange={(selectedOptions) => {
        onChange((selectedOptions as any || []).map((opt: any) => opt.value));
      }}
      placeholder={placeholder || "Search for games..."}
      styles={customStyles}
      className="text-sm"
      noOptionsMessage={({ inputValue }) => 
        inputValue.length < 2 ? "Type at least 2 characters to search..." : "No games found"
      }
      menuPortalTarget={document.body}
    />
  );
};

export default SeedSelector;
