import React from 'react';
import Select from 'react-select';

interface GenreSelectorProps {
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
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
    minHeight: '38px',
  }),
  menu: (base: any) => ({
    ...base,
    backgroundColor: '#1e293b',
    border: '1px solid #334155',
    zIndex: 9999,
    width: '300px', // Allow menu to be wider than the narrow sidebar container
  }),
  menuPortal: (base: any) => ({
    ...base,
    zIndex: 9999,
  }),
  menuList: (base: any) => ({
    ...base,
    maxHeight: '400px', // Show more options at once
    padding: '4px',
    '&::-webkit-scrollbar': {
      width: '4px',
    },
    '&::-webkit-scrollbar-track': {
      background: 'transparent',
    },
    '&::-webkit-scrollbar-thumb': {
      background: '#334155',
      borderres: '10px',
    },
  }),
  option: (base: any, state: any) => ({
    ...base,
    backgroundColor: state.isFocused ? '#334155' : 'transparent',
    color: state.isSelected ? '#eab308' : '#f8fafc',
    padding: '8px 12px',
    fontSize: '12px',
    cursor: 'pointer',
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
    fontSize: '11px',
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
};

const GenreSelector: React.FC<GenreSelectorProps> = ({ options, selected, onChange }) => {
  const selectOptions = options.map(opt => ({ value: opt, label: opt }));
  const value = selected.map(s => ({ value: s, label: s }));

  return (
    <Select
      isMulti
      options={selectOptions}
      value={value}
      onChange={(selectedOptions) => {
        onChange((selectedOptions as any || []).map((opt: any) => opt.value));
      }}
      placeholder="Filter by genres..."
      styles={customStyles}
      className="text-sm"
      menuPortalTarget={document.body}
      blurInputOnSelect={false}
      closeMenuOnSelect={false}
    />
  );
};

export default GenreSelector;
